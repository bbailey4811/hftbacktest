use std::{
    collections::HashMap,
    io::{Error as IoError, ErrorKind},
    ops::{Deref, DerefMut},
};

use data::Reader;
pub use data::{Data, DataSource, read_npz_file};
use models::FeeModel;
use thiserror::Error;

pub use crate::backtest::{
    models::L3QueueModel,
    proc::{L3Local, L3NoPartialFillExchange},
};
use crate::{
    backtest::{
        assettype::AssetType,
        data::{FeedLatencyAdjustment, NpyDTyped},
        evs::{EventIntentKind, EventSet},
        models::{LatencyModel, QueueModel},
        order::order_bus,
        proc::{Local, LocalProcessor, NoPartialFillExchange, PartialFillExchange, Processor},
        state::State,
    },
    depth::{L2MarketDepth, L3MarketDepth, MarketDepth},
    prelude::{
        Bot, OrdType, Order, OrderId, OrderRequest, Side, StateValues, TimeInForce,
        UNTIL_END_OF_DATA, WaitOrderResponse,
    },
    types::{BuildError, ElapseResult, Event},
};

/// Provides asset types.
pub mod assettype;

pub mod models;

/// OrderBus implementation
pub mod order;

/// Local and exchange models
pub mod proc;

/// Trading state.
pub mod state;

/// Recorder for a bot's trading statistics.
pub mod recorder;

pub mod data;
mod evs;

/// Errors that can occur during backtesting.
#[derive(Error, Debug)]
pub enum BacktestError {
    #[error("Order related to a given order id already exists")]
    OrderIdExist,
    #[error("Order request is in process")]
    OrderRequestInProcess,
    #[error("Order not found")]
    OrderNotFound,
    #[error("order request is invalid")]
    InvalidOrderRequest,
    #[error("order status is invalid to proceed the request")]
    InvalidOrderStatus,
    #[error("end of data")]
    EndOfData,
    #[error("data error: {0:?}")]
    DataError(#[from] IoError),
}

/// Backtesting Asset
pub struct Asset<L: ?Sized, E: ?Sized, D: NpyDTyped + Clone /* todo: ugly bounds */> {
    pub local: Box<L>,
    pub exch: Box<E>,
    pub reader: Reader<D>,
}

impl<L, E, D: NpyDTyped + Clone> Asset<L, E, D> {
    /// Constructs an instance of `Asset`. Use this method if a custom local processor or an
    /// exchange processor is needed.
    pub fn new(local: L, exch: E, reader: Reader<D>) -> Self {
        Self {
            local: Box::new(local),
            exch: Box::new(exch),
            reader,
        }
    }

    /// Returns an `L2AssetBuilder`.
    pub fn l2_builder<LM, AT, QM, MD, FM>() -> L2AssetBuilder<LM, AT, QM, MD, FM>
    where
        AT: AssetType + Clone + 'static,
        MD: MarketDepth + L2MarketDepth + 'static,
        QM: QueueModel<MD> + 'static,
        LM: LatencyModel + Clone + 'static,
        FM: FeeModel + Clone + 'static,
    {
        L2AssetBuilder::new()
    }

    /// Returns an `L3AssetBuilder`.
    pub fn l3_builder<LM, AT, QM, MD, FM>() -> L3AssetBuilder<LM, AT, QM, MD, FM>
    where
        AT: AssetType + Clone + 'static,
        MD: MarketDepth + L3MarketDepth + 'static,
        QM: L3QueueModel<MD> + 'static,
        LM: LatencyModel + Clone + 'static,
        FM: FeeModel + Clone + 'static,
        BacktestError: From<<MD as L3MarketDepth>::Error>,
    {
        L3AssetBuilder::new()
    }
}

/// Exchange model kind.
pub enum ExchangeKind {
    /// Uses [NoPartialFillExchange](`NoPartialFillExchange`).
    NoPartialFillExchange,
    /// Uses [PartialFillExchange](`PartialFillExchange`).
    PartialFillExchange,
}

/// A level-2 asset builder.
pub struct L2AssetBuilder<LM, AT, QM, MD, FM> {
    latency_model: Option<LM>,
    asset_type: Option<AT>,
    data: Vec<DataSource<Event>>,
    parallel_load: bool,
    latency_offset: i64,
    fee_model: Option<FM>,
    exch_kind: ExchangeKind,
    last_trades_cap: usize,
    queue_model: Option<QM>,
    depth_builder: Option<Box<dyn Fn() -> MD>>,
    own_fill_depletion: bool,
    own_fill_depletion_persists: bool,
}

impl<LM, AT, QM, MD, FM> L2AssetBuilder<LM, AT, QM, MD, FM>
where
    AT: AssetType + Clone + 'static,
    MD: MarketDepth + L2MarketDepth + 'static,
    QM: QueueModel<MD> + 'static,
    LM: LatencyModel + Clone + 'static,
    FM: FeeModel + Clone + 'static,
{
    /// Constructs an `L2AssetBuilder`.
    pub fn new() -> Self {
        Self {
            latency_model: None,
            asset_type: None,
            data: vec![],
            parallel_load: false,
            latency_offset: 0,
            fee_model: None,
            exch_kind: ExchangeKind::NoPartialFillExchange,
            last_trades_cap: 0,
            queue_model: None,
            depth_builder: None,
            own_fill_depletion: false,
            own_fill_depletion_persists: false,
        }
    }

    /// Sets the feed data.
    pub fn data(self, data: Vec<DataSource<Event>>) -> Self {
        Self { data, ..self }
    }

    /// Sets whether to load the next data in parallel with backtesting. This can speed up the
    /// backtest by reducing data loading time, but it also increases memory usage.
    /// The default value is `true`.
    pub fn parallel_load(self, parallel_load: bool) -> Self {
        Self {
            parallel_load,
            ..self
        }
    }

    /// Sets the latency offset to adjust the feed latency by the specified amount. This is
    /// particularly useful in cross-exchange backtesting, where the feed data is collected from a
    /// different site than the one where the strategy is intended to run.
    pub fn latency_offset(self, latency_offset: i64) -> Self {
        Self {
            latency_offset,
            ..self
        }
    }

    /// Sets a latency model.
    pub fn latency_model(self, latency_model: LM) -> Self {
        Self {
            latency_model: Some(latency_model),
            ..self
        }
    }

    /// Sets an asset type.
    pub fn asset_type(self, asset_type: AT) -> Self {
        Self {
            asset_type: Some(asset_type),
            ..self
        }
    }

    /// Sets a fee model.
    pub fn fee_model(self, fee_model: FM) -> Self {
        Self {
            fee_model: Some(fee_model),
            ..self
        }
    }

    /// Sets an exchange model. The default value is [`NoPartialFillExchange`].
    pub fn exchange(self, exch_kind: ExchangeKind) -> Self {
        Self { exch_kind, ..self }
    }

    /// Sets the initial capacity of the vector storing the last market trades.
    /// The default value is `0`, indicating that no last trades are stored.
    pub fn last_trades_capacity(self, capacity: usize) -> Self {
        Self {
            last_trades_cap: capacity,
            ..self
        }
    }

    /// Sets a queue model.
    pub fn queue_model(self, queue_model: QM) -> Self {
        Self {
            queue_model: Some(queue_model),
            ..self
        }
    }

    /// Sets a market depth builder.
    pub fn depth<Builder>(self, builder: Builder) -> Self
    where
        Builder: Fn() -> MD + 'static,
    {
        Self {
            depth_builder: Some(Box::new(builder)),
            ..self
        }
    }

    /// Sets whether the [`PartialFillExchange`] depletes displayed depth by our own IOC taker fills until the feed
    /// next updates that price level (see [`Self::own_fill_depletion_persists`] for the rule that keeps it longer).
    /// Replayed depth never reflects our own fills, so without depletion a later taker order can fill again against
    /// shares an earlier one already took. The default value is `false`.
    pub fn own_fill_depletion(self, own_fill_depletion: bool) -> Self {
        Self {
            own_fill_depletion,
            ..self
        }
    }

    /// With own-fill depletion enabled, sets whether what our own IOC taker fills took from a price level stays
    /// taken across feed updates of that level, until the level is empty, instead of being forgotten at the level's
    /// next update. A depth clear event still forgets the whole side. The default value is `false`.
    pub fn own_fill_depletion_persists(self, own_fill_depletion_persists: bool) -> Self {
        Self {
            own_fill_depletion_persists,
            ..self
        }
    }

    /// Builds an `Asset`.
    pub fn build(self) -> Result<Asset<dyn LocalProcessor<MD>, dyn Processor, Event>, BuildError> {
        let reader = if self.latency_offset == 0 {
            Reader::builder()
                .parallel_load(self.parallel_load)
                .data(self.data)
                .build()
                .map_err(|err| BuildError::Error(err.into()))?
        } else {
            Reader::builder()
                .parallel_load(self.parallel_load)
                .data(self.data)
                .preprocessor(FeedLatencyAdjustment::new(self.latency_offset))
                .build()
                .map_err(|err| BuildError::Error(err.into()))?
        };

        let create_depth = self
            .depth_builder
            .as_ref()
            .ok_or(BuildError::BuilderIncomplete("depth"))?;
        let order_latency = self
            .latency_model
            .clone()
            .ok_or(BuildError::BuilderIncomplete("order_latency"))?;
        let asset_type = self
            .asset_type
            .clone()
            .ok_or(BuildError::BuilderIncomplete("asset_type"))?;
        let fee_model = self
            .fee_model
            .clone()
            .ok_or(BuildError::BuilderIncomplete("fee_model"))?;

        let (order_e2l, order_l2e) = order_bus(order_latency);

        let local = Local::new(
            create_depth(),
            State::new(asset_type, fee_model),
            self.last_trades_cap,
            order_l2e,
        );

        let queue_model = self
            .queue_model
            .ok_or(BuildError::BuilderIncomplete("queue_model"))?;
        let asset_type = self
            .asset_type
            .clone()
            .ok_or(BuildError::BuilderIncomplete("asset_type"))?;
        let fee_model = self
            .fee_model
            .clone()
            .ok_or(BuildError::BuilderIncomplete("fee_model"))?;

        match self.exch_kind {
            ExchangeKind::NoPartialFillExchange => {
                let exch = NoPartialFillExchange::new(
                    create_depth(),
                    State::new(asset_type, fee_model),
                    queue_model,
                    order_e2l,
                );

                Ok(Asset {
                    local: Box::new(local),
                    exch: Box::new(exch),
                    reader,
                })
            }
            ExchangeKind::PartialFillExchange => {
                let exch = PartialFillExchange::new(
                    create_depth(),
                    State::new(asset_type, fee_model),
                    queue_model,
                    order_e2l,
                )
                .with_own_fill_depletion(self.own_fill_depletion)
                .with_own_fill_depletion_persists(self.own_fill_depletion_persists);

                Ok(Asset {
                    local: Box::new(local),
                    exch: Box::new(exch),
                    reader,
                })
            }
        }
    }
}

impl<LM, AT, QM, MD, FM> Default for L2AssetBuilder<LM, AT, QM, MD, FM>
where
    AT: AssetType + Clone + 'static,
    MD: MarketDepth + L2MarketDepth + 'static,
    QM: QueueModel<MD> + 'static,
    LM: LatencyModel + Clone + 'static,
    FM: FeeModel + Clone + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

/// A level-3 asset builder.
pub struct L3AssetBuilder<LM, AT, QM, MD, FM> {
    latency_model: Option<LM>,
    asset_type: Option<AT>,
    data: Vec<DataSource<Event>>,
    parallel_load: bool,
    latency_offset: i64,
    fee_model: Option<FM>,
    exch_kind: ExchangeKind,
    last_trades_cap: usize,
    queue_model: Option<QM>,
    depth_builder: Option<Box<dyn Fn() -> MD>>,
}

impl<LM, AT, QM, MD, FM> L3AssetBuilder<LM, AT, QM, MD, FM>
where
    AT: AssetType + Clone + 'static,
    MD: MarketDepth + L3MarketDepth + 'static,
    QM: L3QueueModel<MD> + 'static,
    LM: LatencyModel + Clone + 'static,
    FM: FeeModel + Clone + 'static,
    BacktestError: From<<MD as L3MarketDepth>::Error>,
{
    /// Constructs an `L3AssetBuilder`.
    pub fn new() -> Self {
        Self {
            latency_model: None,
            asset_type: None,
            data: vec![],
            parallel_load: false,
            latency_offset: 0,
            fee_model: None,
            exch_kind: ExchangeKind::NoPartialFillExchange,
            last_trades_cap: 0,
            queue_model: None,
            depth_builder: None,
        }
    }

    /// Sets the feed data.
    pub fn data(self, data: Vec<DataSource<Event>>) -> Self {
        Self { data, ..self }
    }

    /// Sets whether to load the next data in parallel with backtesting. This can speed up the
    /// backtest by reducing data loading time, but it also increases memory usage.
    /// The default value is `true`.
    pub fn parallel_load(self, parallel_load: bool) -> Self {
        Self {
            parallel_load,
            ..self
        }
    }

    /// Sets the latency offset to adjust the feed latency by the specified amount. This is
    /// particularly useful in cross-exchange backtesting, where the feed data is collected from a
    /// different site than the one where the strategy is intended to run.
    pub fn latency_offset(self, latency_offset: i64) -> Self {
        Self {
            latency_offset,
            ..self
        }
    }

    /// Sets a latency model.
    pub fn latency_model(self, latency_model: LM) -> Self {
        Self {
            latency_model: Some(latency_model),
            ..self
        }
    }

    /// Sets an asset type.
    pub fn asset_type(self, asset_type: AT) -> Self {
        Self {
            asset_type: Some(asset_type),
            ..self
        }
    }

    /// Sets a fee model.
    pub fn fee_model(self, fee_model: FM) -> Self {
        Self {
            fee_model: Some(fee_model),
            ..self
        }
    }

    /// Sets an exchange model. The default value is [`NoPartialFillExchange`].
    pub fn exchange(self, exch_kind: ExchangeKind) -> Self {
        Self { exch_kind, ..self }
    }

    /// Sets the initial capacity of the vector storing the last market trades.
    /// The default value is `0`, indicating that no last trades are stored.
    pub fn last_trades_capacity(self, capacity: usize) -> Self {
        Self {
            last_trades_cap: capacity,
            ..self
        }
    }

    /// Sets a queue model.
    pub fn queue_model(self, queue_model: QM) -> Self {
        Self {
            queue_model: Some(queue_model),
            ..self
        }
    }

    /// Sets a market depth builder.
    pub fn depth<Builder>(self, builder: Builder) -> Self
    where
        Builder: Fn() -> MD + 'static,
    {
        Self {
            depth_builder: Some(Box::new(builder)),
            ..self
        }
    }

    /// Builds an `Asset`.
    pub fn build(self) -> Result<Asset<dyn LocalProcessor<MD>, dyn Processor, Event>, BuildError> {
        let reader = if self.latency_offset == 0 {
            Reader::builder()
                .parallel_load(self.parallel_load)
                .data(self.data)
                .build()
                .map_err(|err| BuildError::Error(err.into()))?
        } else {
            Reader::builder()
                .parallel_load(self.parallel_load)
                .data(self.data)
                .preprocessor(FeedLatencyAdjustment::new(self.latency_offset))
                .build()
                .map_err(|err| BuildError::Error(err.into()))?
        };

        let create_depth = self
            .depth_builder
            .as_ref()
            .ok_or(BuildError::BuilderIncomplete("depth"))?;
        let order_latency = self
            .latency_model
            .clone()
            .ok_or(BuildError::BuilderIncomplete("order_latency"))?;
        let asset_type = self
            .asset_type
            .clone()
            .ok_or(BuildError::BuilderIncomplete("asset_type"))?;
        let fee_model = self
            .fee_model
            .clone()
            .ok_or(BuildError::BuilderIncomplete("fee_model"))?;

        let (order_e2l, order_l2e) = order_bus(order_latency);

        let local = L3Local::new(
            create_depth(),
            State::new(asset_type, fee_model),
            self.last_trades_cap,
            order_l2e,
        );

        let queue_model = self
            .queue_model
            .ok_or(BuildError::BuilderIncomplete("queue_model"))?;
        let asset_type = self
            .asset_type
            .clone()
            .ok_or(BuildError::BuilderIncomplete("asset_type"))?;
        let fee_model = self
            .fee_model
            .clone()
            .ok_or(BuildError::BuilderIncomplete("fee_model"))?;

        match self.exch_kind {
            ExchangeKind::NoPartialFillExchange => {
                let exch = L3NoPartialFillExchange::new(
                    create_depth(),
                    State::new(asset_type, fee_model),
                    queue_model,
                    order_e2l,
                );

                Ok(Asset {
                    local: Box::new(local),
                    exch: Box::new(exch),
                    reader,
                })
            }
            ExchangeKind::PartialFillExchange => {
                unimplemented!();
            }
        }
    }
}

impl<LM, AT, QM, MD, FM> Default for L3AssetBuilder<LM, AT, QM, MD, FM>
where
    AT: AssetType + Clone + 'static,
    MD: MarketDepth + L3MarketDepth + 'static,
    QM: L3QueueModel<MD> + 'static,
    LM: LatencyModel + Clone + 'static,
    FM: FeeModel + Clone + 'static,
    BacktestError: From<<MD as L3MarketDepth>::Error>,
{
    fn default() -> Self {
        Self::new()
    }
}

/// [`Backtest`] builder.
pub struct BacktestBuilder<MD> {
    local: Vec<BacktestProcessorState<Box<dyn LocalProcessor<MD>>>>,
    exch: Vec<BacktestProcessorState<Box<dyn Processor>>>,
}

impl<MD> BacktestBuilder<MD> {
    /// Adds [`Asset`], which will undergo simulation within the backtester.
    pub fn add_asset(self, asset: Asset<dyn LocalProcessor<MD>, dyn Processor, Event>) -> Self {
        let mut self_ = Self { ..self };
        self_.local.push(BacktestProcessorState::new(
            asset.local,
            asset.reader.clone(),
        ));
        self_
            .exch
            .push(BacktestProcessorState::new(asset.exch, asset.reader));
        self_
    }

    /// Builds [`Backtest`].
    pub fn build(self) -> Result<Backtest<MD>, BuildError> {
        let num_assets = self.local.len();
        if self.local.len() != num_assets || self.exch.len() != num_assets {
            panic!();
        }
        Ok(Backtest {
            cur_ts: i64::MAX,
            evs: EventSet::new(num_assets),
            local: self.local,
            exch: self.exch,
        })
    }
}

/// This backtester provides multi-asset and multi-exchange model backtesting, allowing you to
/// configure different setups such as queue models or asset types for each asset. However, this may
/// result in slightly slower performance compared to [`Backtest`].
pub struct Backtest<MD> {
    cur_ts: i64,
    evs: EventSet,
    local: Vec<BacktestProcessorState<Box<dyn LocalProcessor<MD>>>>,
    exch: Vec<BacktestProcessorState<Box<dyn Processor>>>,
}

impl<P: Processor> Deref for BacktestProcessorState<P> {
    type Target = P;

    fn deref(&self) -> &Self::Target {
        &self.processor
    }
}

impl<P: Processor> DerefMut for BacktestProcessorState<P> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.processor
    }
}

/// Per asset backtesting state used internally to advance event buffers.
pub struct BacktestProcessorState<P: Processor> {
    data: Data<Event>,
    processor: P,
    reader: Reader<Event>,
    row: Option<usize>,
}

impl<P: Processor> BacktestProcessorState<P> {
    fn new(processor: P, reader: Reader<Event>) -> BacktestProcessorState<P> {
        Self {
            data: Data::empty(),
            processor,
            reader,
            row: None,
        }
    }

    /// Get the index of the next available row, only advancing the reader if there's no
    /// row currently available.
    fn next_row(&mut self) -> Result<usize, BacktestError> {
        if self.row.is_none() {
            let _ = self.advance()?;
        }

        self.row.ok_or(BacktestError::EndOfData)
    }

    /// Advance the state of this processor to the next available event and return the
    /// timestamp it occurred at, if any.
    fn advance(&mut self) -> Result<i64, BacktestError> {
        loop {
            let start = self.row.map(|rn| rn + 1).unwrap_or(0);

            for rn in start..self.data.len() {
                if let Some(ts) = self.processor.event_seen_timestamp(&self.data[rn]) {
                    self.row = Some(rn);
                    return Ok(ts);
                }
            }

            let next = self.reader.next_data()?;

            self.reader.release(std::mem::replace(&mut self.data, next));
            self.row = None;
        }
    }
}

impl<MD> Backtest<MD>
where
    MD: MarketDepth,
{
    pub fn builder() -> BacktestBuilder<MD> {
        BacktestBuilder {
            local: vec![],
            exch: vec![],
        }
    }

    pub fn new(
        local: Vec<Box<dyn LocalProcessor<MD>>>,
        exch: Vec<Box<dyn Processor>>,
        reader: Vec<Reader<Event>>,
    ) -> Self {
        let num_assets = local.len();
        if local.len() != num_assets || exch.len() != num_assets || reader.len() != num_assets {
            panic!();
        }

        let local = local
            .into_iter()
            .zip(reader.iter())
            .map(|(proc, reader)| BacktestProcessorState::new(proc, reader.clone()))
            .collect();
        let exch = exch
            .into_iter()
            .zip(reader.iter())
            .map(|(proc, reader)| BacktestProcessorState::new(proc, reader.clone()))
            .collect();

        Self {
            local,
            exch,
            cur_ts: i64::MAX,
            evs: EventSet::new(num_assets),
        }
    }

    fn initialize_evs(&mut self) -> Result<(), BacktestError> {
        for (asset_no, local) in self.local.iter_mut().enumerate() {
            match local.advance() {
                Ok(ts) => self.evs.update_local_data(asset_no, ts),
                Err(BacktestError::EndOfData) => {
                    self.evs.invalidate_local_data(asset_no);
                }
                Err(e) => {
                    return Err(e);
                }
            }
        }
        for (asset_no, exch) in self.exch.iter_mut().enumerate() {
            match exch.advance() {
                Ok(ts) => self.evs.update_exch_data(asset_no, ts),
                Err(BacktestError::EndOfData) => {
                    self.evs.invalidate_exch_data(asset_no);
                }
                Err(e) => {
                    return Err(e);
                }
            }
        }
        Ok(())
    }

    pub fn goto_end(&mut self) -> Result<ElapseResult, BacktestError> {
        if self.cur_ts == i64::MAX {
            self.initialize_evs()?;
            match self.evs.next() {
                Some(ev) => {
                    self.cur_ts = ev.timestamp;
                }
                None => {
                    return Ok(ElapseResult::EndOfData);
                }
            }
        }
        self.goto::<false>(UNTIL_END_OF_DATA, WaitOrderResponse::None)
    }

    /// Applies one externally supplied feed event to the local and exchange
    /// processors after advancing outstanding order timers to the event time.
    ///
    /// This is intended for live/protocol simulation adapters that receive
    /// market data from an external websocket instead of a static [`Reader`].
    /// It deliberately reuses the same processors, order queues, fee model,
    /// latency model, and market-depth logic as normal backtests; only the
    /// feed source is external.
    pub fn apply_external_feed_event(
        &mut self,
        asset_no: usize,
        event: &Event,
    ) -> Result<ElapseResult, BacktestError> {
        if self.cur_ts == i64::MAX {
            self.initialize_evs()?;
            match self.evs.next() {
                Some(ev) => {
                    self.cur_ts = ev.timestamp;
                }
                None => {
                    self.cur_ts = 0;
                }
            }
        }

        let local_ts = self
            .local
            .get(asset_no)
            .and_then(|local| local.event_seen_timestamp(event));
        let exch_ts = self
            .exch
            .get(asset_no)
            .and_then(|exch| exch.event_seen_timestamp(event));
        let Some(target_ts) = local_ts.into_iter().chain(exch_ts).max() else {
            return Ok(ElapseResult::Ok);
        };

        let mut result = ElapseResult::Ok;
        if target_ts > self.cur_ts {
            result = self.goto::<false>(target_ts, WaitOrderResponse::None)?;
        }

        if let Some(ts) = local_ts {
            if ts < self.cur_ts {
                return Err(BacktestError::DataError(IoError::new(
                    ErrorKind::InvalidData,
                    "external local feed event timestamp moved backwards",
                )));
            }
            let local = self.local.get_mut(asset_no).unwrap();
            local.processor.process(event)?;
            self.cur_ts = ts;
        }
        if let Some(ts) = exch_ts {
            if ts < self.cur_ts {
                return Err(BacktestError::DataError(IoError::new(
                    ErrorKind::InvalidData,
                    "external exchange feed event timestamp moved backwards",
                )));
            }
            let exch = self.exch.get_mut(asset_no).unwrap();
            exch.processor.process(event)?;
            self.cur_ts = ts;
        }
        Ok(result)
    }

    fn goto<const WAIT_NEXT_FEED: bool>(
        &mut self,
        timestamp: i64,
        wait_order_response: WaitOrderResponse,
    ) -> Result<ElapseResult, BacktestError> {
        let mut result = ElapseResult::Ok;
        let mut timestamp = timestamp;
        for (asset_no, local) in self.local.iter().enumerate() {
            self.evs
                .update_exch_order(asset_no, local.earliest_send_order_timestamp());
            self.evs
                .update_local_order(asset_no, local.earliest_recv_order_timestamp());
        }
        loop {
            match self.evs.next() {
                Some(ev) => {
                    if ev.timestamp > timestamp {
                        self.cur_ts = timestamp;
                        return Ok(result);
                    }
                    match ev.kind {
                        EventIntentKind::LocalData => {
                            let local = unsafe { self.local.get_unchecked_mut(ev.asset_no) };
                            let next = local.next_row().and_then(|row| {
                                local.processor.process(&local.data[row])?;
                                local.advance()
                            });

                            match next {
                                Ok(next_ts) => {
                                    self.evs.update_local_data(ev.asset_no, next_ts);
                                }
                                Err(BacktestError::EndOfData) => {
                                    self.evs.invalidate_local_data(ev.asset_no);
                                }
                                Err(e) => {
                                    return Err(e);
                                }
                            }
                            if WAIT_NEXT_FEED {
                                timestamp = ev.timestamp;
                                result = ElapseResult::MarketFeed;
                            }
                        }
                        EventIntentKind::LocalOrder => {
                            let local = unsafe { self.local.get_unchecked_mut(ev.asset_no) };
                            let wait_order_resp_id = match wait_order_response {
                                WaitOrderResponse::Specified {
                                    asset_no: wait_order_asset_no,
                                    order_id: wait_order_id,
                                } if ev.asset_no == wait_order_asset_no => Some(wait_order_id),
                                _ => None,
                            };
                            if local.process_recv_order(ev.timestamp, wait_order_resp_id)?
                                || wait_order_response == WaitOrderResponse::Any
                            {
                                timestamp = ev.timestamp;
                                if WAIT_NEXT_FEED {
                                    result = ElapseResult::OrderResponse;
                                }
                            }
                            self.evs.update_local_order(
                                ev.asset_no,
                                local.earliest_recv_order_timestamp(),
                            );
                        }
                        EventIntentKind::ExchData => {
                            let exch = unsafe { self.exch.get_unchecked_mut(ev.asset_no) };
                            let next = exch.next_row().and_then(|row| {
                                exch.processor.process(&exch.data[row])?;
                                exch.advance()
                            });

                            match next {
                                Ok(next_ts) => {
                                    self.evs.update_exch_data(ev.asset_no, next_ts);
                                }
                                Err(BacktestError::EndOfData) => {
                                    self.evs.invalidate_exch_data(ev.asset_no);
                                }
                                Err(e) => {
                                    return Err(e);
                                }
                            }
                            self.evs.update_local_order(
                                ev.asset_no,
                                exch.earliest_send_order_timestamp(),
                            );
                        }
                        EventIntentKind::ExchOrder => {
                            let exch = unsafe { self.exch.get_unchecked_mut(ev.asset_no) };
                            let _ = exch.process_recv_order(ev.timestamp, None)?;
                            self.evs.update_exch_order(
                                ev.asset_no,
                                exch.earliest_recv_order_timestamp(),
                            );
                            self.evs.update_local_order(
                                ev.asset_no,
                                exch.earliest_send_order_timestamp(),
                            );
                        }
                    }
                }
                None => {
                    return Ok(ElapseResult::EndOfData);
                }
            }
        }
    }
}

impl<MD> Bot<MD> for Backtest<MD>
where
    MD: MarketDepth,
{
    type Error = BacktestError;

    #[inline]
    fn current_timestamp(&self) -> i64 {
        self.cur_ts
    }

    #[inline]
    fn num_assets(&self) -> usize {
        self.local.len()
    }

    #[inline]
    fn position(&self, asset_no: usize) -> f64 {
        self.local.get(asset_no).unwrap().position()
    }

    #[inline]
    fn state_values(&self, asset_no: usize) -> &StateValues {
        self.local.get(asset_no).unwrap().state_values()
    }

    fn depth(&self, asset_no: usize) -> &MD {
        self.local.get(asset_no).unwrap().depth()
    }

    fn last_trades(&self, asset_no: usize) -> &[Event] {
        self.local.get(asset_no).unwrap().last_trades()
    }

    #[inline]
    fn clear_last_trades(&mut self, asset_no: Option<usize>) {
        match asset_no {
            Some(an) => {
                let local = self.local.get_mut(an).unwrap();
                local.clear_last_trades();
            }
            None => {
                for local in self.local.iter_mut() {
                    local.clear_last_trades();
                }
            }
        }
    }

    #[inline]
    fn orders(&self, asset_no: usize) -> &HashMap<u64, Order> {
        self.local.get(asset_no).unwrap().orders()
    }

    #[inline]
    fn submit_buy_order(
        &mut self,
        asset_no: usize,
        order_id: OrderId,
        price: f64,
        qty: f64,
        time_in_force: TimeInForce,
        order_type: OrdType,
        wait: bool,
    ) -> Result<ElapseResult, Self::Error> {
        let local = self.local.get_mut(asset_no).unwrap();
        local.submit_order(
            order_id,
            Side::Buy,
            price,
            qty,
            order_type,
            time_in_force,
            self.cur_ts,
        )?;

        if wait {
            return self.goto::<false>(
                UNTIL_END_OF_DATA,
                WaitOrderResponse::Specified { asset_no, order_id },
            );
        }
        Ok(ElapseResult::Ok)
    }

    #[inline]
    fn submit_sell_order(
        &mut self,
        asset_no: usize,
        order_id: OrderId,
        price: f64,
        qty: f64,
        time_in_force: TimeInForce,
        order_type: OrdType,
        wait: bool,
    ) -> Result<ElapseResult, Self::Error> {
        let local = self.local.get_mut(asset_no).unwrap();
        local.submit_order(
            order_id,
            Side::Sell,
            price,
            qty,
            order_type,
            time_in_force,
            self.cur_ts,
        )?;

        if wait {
            return self.goto::<false>(
                UNTIL_END_OF_DATA,
                WaitOrderResponse::Specified { asset_no, order_id },
            );
        }
        Ok(ElapseResult::Ok)
    }

    fn submit_order(
        &mut self,
        asset_no: usize,
        order: OrderRequest,
        wait: bool,
    ) -> Result<ElapseResult, Self::Error> {
        let local = self.local.get_mut(asset_no).unwrap();
        local.submit_order(
            order.order_id,
            order.side,
            order.price,
            order.qty,
            order.order_type,
            order.time_in_force,
            self.cur_ts,
        )?;

        if wait {
            return self.goto::<false>(
                UNTIL_END_OF_DATA,
                WaitOrderResponse::Specified {
                    asset_no,
                    order_id: order.order_id,
                },
            );
        }
        Ok(ElapseResult::Ok)
    }

    #[inline]
    fn modify(
        &mut self,
        asset_no: usize,
        order_id: OrderId,
        price: f64,
        qty: f64,
        wait: bool,
    ) -> Result<ElapseResult, Self::Error> {
        let local = self.local.get_mut(asset_no).unwrap();
        local.modify(order_id, price, qty, self.cur_ts)?;

        if wait {
            return self.goto::<false>(
                UNTIL_END_OF_DATA,
                WaitOrderResponse::Specified { asset_no, order_id },
            );
        }
        Ok(ElapseResult::Ok)
    }

    #[inline]
    fn cancel(
        &mut self,
        asset_no: usize,
        order_id: OrderId,
        wait: bool,
    ) -> Result<ElapseResult, Self::Error> {
        let local = self.local.get_mut(asset_no).unwrap();
        local.cancel(order_id, self.cur_ts)?;

        if wait {
            return self.goto::<false>(
                UNTIL_END_OF_DATA,
                WaitOrderResponse::Specified { asset_no, order_id },
            );
        }
        Ok(ElapseResult::Ok)
    }

    #[inline]
    fn clear_inactive_orders(&mut self, asset_no: Option<usize>) {
        match asset_no {
            Some(asset_no) => {
                self.local
                    .get_mut(asset_no)
                    .unwrap()
                    .clear_inactive_orders();
            }
            None => {
                for local in self.local.iter_mut() {
                    local.clear_inactive_orders();
                }
            }
        }
    }

    #[inline]
    fn wait_order_response(
        &mut self,
        asset_no: usize,
        order_id: OrderId,
        timeout: i64,
    ) -> Result<ElapseResult, BacktestError> {
        self.goto::<false>(
            self.cur_ts + timeout,
            WaitOrderResponse::Specified { asset_no, order_id },
        )
    }

    #[inline]
    fn wait_next_feed(
        &mut self,
        include_order_resp: bool,
        timeout: i64,
    ) -> Result<ElapseResult, Self::Error> {
        if self.cur_ts == i64::MAX {
            self.initialize_evs()?;
            match self.evs.next() {
                Some(ev) => {
                    self.cur_ts = ev.timestamp;
                }
                None => {
                    return Ok(ElapseResult::EndOfData);
                }
            }
        }
        if include_order_resp {
            self.goto::<true>(self.cur_ts + timeout, WaitOrderResponse::Any)
        } else {
            self.goto::<true>(self.cur_ts + timeout, WaitOrderResponse::None)
        }
    }

    #[inline]
    fn elapse(&mut self, duration: i64) -> Result<ElapseResult, Self::Error> {
        if self.cur_ts == i64::MAX {
            self.initialize_evs()?;
            match self.evs.next() {
                Some(ev) => {
                    self.cur_ts = ev.timestamp;
                }
                None => {
                    return Ok(ElapseResult::EndOfData);
                }
            }
        }
        self.goto::<false>(self.cur_ts + duration, WaitOrderResponse::None)
    }

    #[inline]
    fn elapse_bt(&mut self, duration: i64) -> Result<ElapseResult, Self::Error> {
        self.elapse(duration)
    }

    #[inline]
    fn close(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }

    #[inline]
    fn feed_latency(&self, asset_no: usize) -> Option<(i64, i64)> {
        self.local.get(asset_no).unwrap().feed_latency()
    }

    #[inline]
    fn order_latency(&self, asset_no: usize) -> Option<(i64, i64, i64)> {
        self.local.get(asset_no).unwrap().order_latency()
    }
}

#[cfg(test)]
mod test {
    use std::error::Error;

    use crate::{
        backtest::{
            Backtest, DataSource,
            ExchangeKind::NoPartialFillExchange,
            L2AssetBuilder,
            assettype::LinearAsset,
            data::Data,
            models::{
                CommonFees, ConstantLatency, PowerProbQueueFunc3, ProbQueueModel,
                TradingValueFeeModel,
            },
        },
        depth::HashMapMarketDepth,
        prelude::{Bot, Event},
        types::{EXCH_EVENT, LOCAL_EVENT},
    };

    #[test]
    fn skips_unseen_events() -> Result<(), Box<dyn Error>> {
        let data = Data::from_data(&[
            Event {
                ev: EXCH_EVENT | LOCAL_EVENT,
                exch_ts: 0,
                local_ts: 0,
                px: 0.0,
                qty: 0.0,
                order_id: 0,
                ival: 0,
                fval: 0.0,
            },
            Event {
                ev: LOCAL_EVENT | EXCH_EVENT,
                exch_ts: 1,
                local_ts: 1,
                px: 0.0,
                qty: 0.0,
                order_id: 0,
                ival: 0,
                fval: 0.0,
            },
            Event {
                ev: EXCH_EVENT,
                exch_ts: 3,
                local_ts: 4,
                px: 0.0,
                qty: 0.0,
                order_id: 0,
                ival: 0,
                fval: 0.0,
            },
            Event {
                ev: LOCAL_EVENT,
                exch_ts: 3,
                local_ts: 4,
                px: 0.0,
                qty: 0.0,
                order_id: 0,
                ival: 0,
                fval: 0.0,
            },
        ]);

        let mut backtester = Backtest::builder()
            .add_asset(
                L2AssetBuilder::default()
                    .data(vec![DataSource::Data(data)])
                    .latency_model(ConstantLatency::new(50, 50))
                    .asset_type(LinearAsset::new(1.0))
                    .fee_model(TradingValueFeeModel::new(CommonFees::new(0.0, 0.0)))
                    .queue_model(ProbQueueModel::new(PowerProbQueueFunc3::new(3.0)))
                    .exchange(NoPartialFillExchange)
                    .depth(|| HashMapMarketDepth::new(0.01, 1.0))
                    .build()?,
            )
            .build()?;

        // Process first events and advance a single timestep
        backtester.elapse_bt(1)?;
        assert_eq!(1, backtester.cur_ts);

        // Check that we correctly skip past events that aren't seen by a given processor
        backtester.elapse_bt(1)?;
        assert_eq!(2, backtester.cur_ts);
        assert_eq!(Some(3), backtester.local[0].row);
        assert_eq!(Some(2), backtester.exch[0].row);

        backtester.elapse_bt(1)?;
        assert_eq!(3, backtester.cur_ts);

        Ok(())
    }
}

#[cfg(test)]
mod ab3308_maker_terminal {
    use crate::{
        backtest::{
            Backtest, DataSource, ExchangeKind, L2AssetBuilder,
            assettype::LinearAsset,
            data::Data,
            models::{
                CommonFees, ConstantLatency, PowerProbQueueFunc3, ProbQueueModel,
                TradingValueFeeModel,
            },
        },
        depth::HashMapMarketDepth,
        prelude::Bot,
        types::*,
    };
    #[test]
    fn exactly_exhausted_maker_order_is_not_filled_again_on_either_side() {
        for buy in [true, false] {
            let event = |t, ev, px, qty| Event {
                ev: ev | EXCH_EVENT | LOCAL_EVENT,
                exch_ts: t,
                local_ts: t,
                px,
                qty,
                order_id: 0,
                ival: 0,
                fval: 0.0,
            };
            let price = if buy { 99.0 } else { 101.0 };
            let trade = TRADE_EVENT | if buy { SELL_EVENT } else { BUY_EVENT };
            let data = Data::from_data(&[
                event(1, DEPTH_EVENT | BUY_EVENT, 99.0, 1.0),
                event(2, DEPTH_EVENT | SELL_EVENT, 101.0, 1.0),
                event(3, 0, 0.0, 0.0),
                event(10, trade, price, 2.0),
                event(11, trade, price, 1.0),
                event(20, 0, 0.0, 0.0),
            ]);
            let asset = L2AssetBuilder::default()
                .data(vec![DataSource::Data(data)])
                .latency_model(ConstantLatency::new(0, 0))
                .asset_type(LinearAsset::new(1.0))
                .fee_model(TradingValueFeeModel::new(CommonFees::new(0.00004, 0.00028)))
                .queue_model(ProbQueueModel::new(PowerProbQueueFunc3::new(3.0)))
                .exchange(ExchangeKind::PartialFillExchange)
                .depth(|| HashMapMarketDepth::new(1.0, 1.0))
                .build()
                .unwrap();
            let mut bt = Backtest::builder().add_asset(asset).build().unwrap();
            bt.elapse(3).unwrap();
            if buy {
                bt.submit_buy_order(0, 1, price, 1.0, TimeInForce::GTX, OrdType::Limit, false)
                    .unwrap();
            } else {
                bt.submit_sell_order(0, 1, price, 1.0, TimeInForce::GTX, OrdType::Limit, false)
                    .unwrap();
            }
            bt.elapse(20)
                .expect("AB3308 later trade cannot refill a fully executed maker order");
            assert_eq!(bt.position(0), if buy { 1.0 } else { -1.0 });
            assert!((bt.state_values(0).fee - price * 0.00004).abs() < 1e-12);
        }
    }
}

#[cfg(test)]
mod ab3465_modify_is_not_an_execution {
    use crate::{
        backtest::{
            Backtest, DataSource, ExchangeKind, L2AssetBuilder,
            assettype::LinearAsset,
            data::Data,
            models::{
                CommonFees, ConstantLatency, PowerProbQueueFunc3, ProbQueueModel,
                TradingValueFeeModel,
            },
        },
        depth::HashMapMarketDepth,
        prelude::Bot,
        types::*,
    };

    #[test]
    fn reducing_a_resting_maker_order_books_no_fill() {
        for buy in [true, false] {
            let event = |t, ev, px, qty| Event {
                ev: ev | EXCH_EVENT | LOCAL_EVENT,
                exch_ts: t,
                local_ts: t,
                px,
                qty,
                order_id: 0,
                ival: 0,
                fval: 0.0,
            };
            let price = if buy { 99.0 } else { 101.0 };
            let data = Data::from_data(&[
                event(1, DEPTH_EVENT | BUY_EVENT, 99.0, 1.0),
                event(2, DEPTH_EVENT | SELL_EVENT, 101.0, 1.0),
                event(3, 0, 0.0, 0.0),
                event(30, 0, 0.0, 0.0),
            ]);
            let asset = L2AssetBuilder::default()
                .data(vec![DataSource::Data(data)])
                .latency_model(ConstantLatency::new(1, 1))
                .asset_type(LinearAsset::new(1.0))
                .fee_model(TradingValueFeeModel::new(CommonFees::new(0.00004, 0.00028)))
                .queue_model(ProbQueueModel::new(PowerProbQueueFunc3::new(3.0)))
                .exchange(ExchangeKind::PartialFillExchange)
                .depth(|| HashMapMarketDepth::new(1.0, 1.0))
                .build()
                .unwrap();
            let mut bt = Backtest::builder().add_asset(asset).build().unwrap();
            bt.elapse(3).unwrap();
            if buy {
                bt.submit_buy_order(0, 1, price, 1.0, TimeInForce::GTX, OrdType::Limit, true)
                    .unwrap();
            } else {
                bt.submit_sell_order(0, 1, price, 1.0, TimeInForce::GTX, OrdType::Limit, true)
                    .unwrap();
            }
            bt.modify(0, 1, price, 0.5, true).unwrap();
            bt.elapse(20).unwrap();
            assert_eq!(bt.position(0), 0.0, "a size reduction is not an execution");
            assert_eq!(bt.state_values(0).balance, 0.0);
            assert_eq!(bt.state_values(0).num_trades, 0);
        }
    }
}

#[cfg(test)]
mod ab3465_through_fills_capped {
    use crate::{
        backtest::{
            Backtest, DataSource, ExchangeKind, L2AssetBuilder,
            assettype::LinearAsset,
            data::Data,
            models::{
                CommonFees, ConstantLatency, PowerProbQueueFunc3, ProbQueueModel,
                TradingValueFeeModel,
            },
        },
        depth::HashMapMarketDepth,
        prelude::Bot,
        types::*,
    };

    fn event(t: i64, ev: u64, px: f64, qty: f64) -> Event {
        Event {
            ev: ev | EXCH_EVENT | LOCAL_EVENT,
            exch_ts: t,
            local_ts: t,
            px,
            qty,
            order_id: 0,
            ival: 0,
            fval: 0.0,
        }
    }

    /// Book 98 / 102, our 1.0 maker order at 100 (best on its side) from t=3, then `later` events.
    fn position_after(buy: bool, later: &[Event]) -> f64 {
        let mut events = vec![
            event(1, DEPTH_EVENT | BUY_EVENT, 98.0, 1.0),
            event(2, DEPTH_EVENT | SELL_EVENT, 102.0, 1.0),
            event(3, 0, 0.0, 0.0),
        ];
        events.extend_from_slice(later);
        events.push(event(50, 0, 0.0, 0.0));
        let asset = L2AssetBuilder::default()
            .data(vec![DataSource::Data(Data::from_data(&events))])
            .latency_model(ConstantLatency::new(0, 0))
            .asset_type(LinearAsset::new(1.0))
            .fee_model(TradingValueFeeModel::new(CommonFees::new(0.00004, 0.00028)))
            .queue_model(ProbQueueModel::new(PowerProbQueueFunc3::new(3.0)))
            .exchange(ExchangeKind::PartialFillExchange)
            .depth(|| HashMapMarketDepth::new(1.0, 0.01))
            .build()
            .unwrap();
        let mut bt = Backtest::builder().add_asset(asset).build().unwrap();
        bt.elapse(3).unwrap();
        if buy {
            bt.submit_buy_order(0, 1, 100.0, 1.0, TimeInForce::GTX, OrdType::Limit, false)
                .unwrap();
        } else {
            bt.submit_sell_order(0, 1, 100.0, 1.0, TimeInForce::GTX, OrdType::Limit, false)
                .unwrap();
        }
        bt.elapse(50).unwrap();
        bt.position(0)
    }

    #[test]
    fn a_trade_through_our_price_fills_only_the_quantity_it_traded() {
        for buy in [true, false] {
            let (through, sign) = if buy { (99.0, 1.0) } else { (101.0, -1.0) };
            let taker = TRADE_EVENT | if buy { SELL_EVENT } else { BUY_EVENT };
            let pos = position_after(
                buy,
                &[
                    event(10, taker, through, 0.03),
                    event(11, taker, through, 0.05),
                ],
            );
            assert!(
                (pos - sign * 0.08).abs() < 1e-9,
                "buy={buy}: two prints of 0.03 and 0.05 fill 0.08, got {pos}"
            );
        }
    }

    #[test]
    fn the_book_crossing_our_price_fills_only_the_crossing_quantity_once() {
        for buy in [true, false] {
            let (side, sign, deeper) = if buy {
                (SELL_EVENT, 1.0, 99.0)
            } else {
                (BUY_EVENT, -1.0, 101.0)
            };
            let pos = position_after(
                buy,
                &[
                    // 0.2 rests at our price: it meets our order for 0.2.
                    event(10, DEPTH_EVENT | side, 100.0, 0.2),
                    // 0.1 more through our price: 0.3 now crosses, 0.2 of it already filled us.
                    event(11, DEPTH_EVENT | side, deeper, 0.1),
                ],
            );
            assert!(
                (pos - sign * 0.3).abs() < 1e-9,
                "buy={buy}: crossing quantity 0.3 fills 0.3, got {pos}"
            );
        }
    }
}

#[cfg(test)]
mod ab3465_review_cases {
    use crate::{
        backtest::{
            Backtest, DataSource, ExchangeKind, L2AssetBuilder,
            assettype::LinearAsset,
            data::Data,
            models::{
                CommonFees, ConstantLatency, PowerProbQueueFunc3, ProbQueueModel,
                TradingValueFeeModel,
            },
        },
        depth::{HashMapMarketDepth, MarketDepth},
        prelude::Bot,
        types::*,
    };

    fn ev(t: i64, e: u64, px: f64, qty: f64) -> Event {
        Event { ev: e | EXCH_EVENT | LOCAL_EVENT, exch_ts: t, local_ts: t, px, qty, order_id: 0, ival: 0, fval: 0.0 }
    }
    fn bid(t: i64, px: f64, q: f64) -> Event { ev(t, DEPTH_EVENT | BUY_EVENT, px, q) }
    fn ask(t: i64, px: f64, q: f64) -> Event { ev(t, DEPTH_EVENT | SELL_EVENT, px, q) }
    fn buy_trade(t: i64, px: f64, q: f64) -> Event { ev(t, TRADE_EVENT | BUY_EVENT, px, q) }
    fn sell_trade(t: i64, px: f64, q: f64) -> Event { ev(t, TRADE_EVENT | SELL_EVENT, px, q) }

    /// Book 98 / 102. `orders` = (id, is_buy, px, qty) GTX at t=3, then `later`. Returns (position, open order count).
    fn run(orders: &[(u64, bool, f64, f64)], later: &[Event]) -> (f64, usize) {
        let mut events = vec![bid(1, 98.0, 1.0), ask(2, 102.0, 1.0), ev(3, 0, 0.0, 0.0)];
        events.extend_from_slice(later);
        events.push(ev(50, 0, 0.0, 0.0));
        let asset = L2AssetBuilder::default()
            .data(vec![DataSource::Data(Data::from_data(&events))])
            .latency_model(ConstantLatency::new(0, 0))
            .asset_type(LinearAsset::new(1.0))
            .fee_model(TradingValueFeeModel::new(CommonFees::new(0.00004, 0.00028)))
            .queue_model(ProbQueueModel::new(PowerProbQueueFunc3::new(3.0)))
            .exchange(ExchangeKind::PartialFillExchange)
            .depth(|| HashMapMarketDepth::new(1.0, 0.01))
            .build()
            .unwrap();
        let mut bt = Backtest::builder().add_asset(asset).build().unwrap();
        bt.elapse(3).unwrap();
        for &(id, b, px, q) in orders {
            if b {
                bt.submit_buy_order(0, id, px, q, TimeInForce::GTX, OrdType::Limit, false).unwrap();
            } else {
                bt.submit_sell_order(0, id, px, q, TimeInForce::GTX, OrdType::Limit, false).unwrap();
            }
        }
        bt.elapse(50).unwrap();
        let open = bt.orders(0).values().filter(|o| o.active()).count();
        (bt.position(0), open)
    }

    fn near(a: f64, b: f64) -> bool { (a - b).abs() < 1e-9 }

    // R1: a print through a crossed remainder fills it.
    #[test]
    fn r1_crossed_remainder_trade_through_near() {
        let (p, _) = run(&[(1, false, 100.0, 1.0)], &[bid(10, 101.0, 0.2), buy_trade(11, 102.0, 0.1)]);
        assert!(near(p, -0.3), "print at 102 through our 100 ask should fill 0.1 more; got {p}");
    }
    #[test]
    fn r1b_crossed_remainder_trade_through_far() {
        let (p, _) = run(&[(1, false, 100.0, 1.0)], &[bid(10, 101.0, 0.2), buy_trade(11, 104.0, 0.1)]);
        assert!(near(p, -0.3), "got {p}");
    }
    #[test]
    fn r1c_buy_side_mirror() {
        let (p, _) = run(&[(1, true, 100.0, 1.0)], &[ask(10, 99.0, 0.2), sell_trade(11, 98.0, 0.1)]);
        assert!(near(p, 0.3), "print at 98 through our 100 bid should fill 0.1 more; got {p}");
    }

    // R2: crossing quantity that leaves and comes back fills again.
    #[test]
    fn r2_recross_by_new_liquidity() {
        let (p, _) = run(
            &[(1, false, 100.0, 1.0)],
            &[bid(10, 101.0, 0.2), bid(11, 101.0, 0.0), bid(12, 101.0, 0.2)],
        );
        assert!(near(p, -0.4), "a second 0.2 bid crossing after the first left should fill 0.2; got {p}");
    }

    // R3: crossing quantity that grows without the best moving fills as it grows.
    #[test]
    fn r3_crossing_level_grows() {
        let (p, _) = run(&[(1, false, 100.0, 1.0)], &[bid(10, 101.0, 0.2), bid(11, 101.0, 0.5)]);
        assert!(near(p, -0.5), "crossing grew to 0.5; got {p}");
    }
    #[test]
    fn r3b_backlog_released_by_unrelated_tick() {
        let (p, _) = run(
            &[(1, false, 100.0, 1.0)],
            &[bid(10, 101.0, 0.2), bid(11, 101.0, 0.5), bid(12, 102.0, 0.01)],
        );
        assert!(near(p, -0.51), "got {p}");
    }

    // R4: one print or one crossing level is one budget across all our orders.
    #[test]
    fn r4_one_print_fills_two_own_orders() {
        let (p, _) = run(&[(1, false, 100.0, 1.0), (2, false, 101.0, 1.0)], &[buy_trade(10, 102.0, 0.03)]);
        assert!(near(p, -0.03), "a 0.03 taker fills 0.03 across our book; got {p}");
    }
    #[test]
    fn r4b_one_crossing_level_fills_two_own_orders() {
        let (p, _) = run(&[(1, false, 100.0, 1.0), (2, false, 101.0, 1.0)], &[bid(10, 101.0, 0.2)]);
        assert!(near(p, -0.2), "a 0.2 crossing bid fills 0.2 across our book; got {p}");
    }

    // R5: through-trade then a cross from the same seller's rest: should equal seller size (sanity, expected pass).
    #[test]
    fn r5_taker_rest_after_through_print() {
        let (p, _) = run(&[(1, true, 100.0, 1.0)], &[sell_trade(10, 98.0, 0.3), bid(11, 98.0, 0.7), ask(12, 98.0, 0.2)]);
        assert!(near(p, 0.5), "got {p}");
    }

    // R6: depth clear + snapshot re-send does not double-fill (expected pass).
    #[test]
    fn r6_clear_and_resnapshot() {
        let (p, _) = run(
            &[(1, false, 100.0, 1.0)],
            &[bid(10, 101.0, 0.2), ev(11, DEPTH_CLEAR_EVENT | BUY_EVENT, 0.0, 0.0), bid(12, 98.0, 1.0), bid(12, 101.0, 0.2)],
        );
        assert!(near(p, -0.2), "got {p}");
    }

    // R7: a zero-quantity print through our price.
    #[test]
    fn r7_zero_qty_print() {
        let (p, open) = run(&[(1, false, 100.0, 1.0)], &[buy_trade(10, 102.0, 0.0)]);
        assert!(near(p, 0.0) && open == 1, "got {p} open={open}");
    }

    // R8: sum of many prints exactly fills and retires the order; extra prints do not panic.
    #[test]
    fn r8_exact_fill_then_more_prints() {
        let mut later = vec![];
        for i in 0..12 { later.push(buy_trade(10 + i, 102.0, 0.1)); }
        let (p, open) = run(&[(1, false, 100.0, 1.0)], &later);
        assert!(near(p, -1.0) && open == 0, "got {p} open={open}");
    }

    // R9: cross partial + trade-through completes and retires; then further crossing is ignored.
    #[test]
    fn r9_cross_then_through_then_cross() {
        let (p, open) = run(
            &[(1, false, 100.0, 1.0)],
            &[bid(10, 101.0, 0.6), buy_trade(11, 106.0, 0.4), bid(12, 101.0, 2.0), bid(13, 103.0, 2.0)],
        );
        assert!(near(p, -1.0) && open == 0, "got {p} open={open}");
    }

    // R10: only displayed crossing quantity fills.
    #[test]
    fn r10_stale_level_counted_as_crossing() {
        // bids 98(1.0) + 101(0.5) is set up after the base book; ask 100 crosses 101 (left stale), then ask 100 leaves.
        let (p, _) = run(
            &[(1, false, 100.0, 1.0)],
            &[bid(10, 99.0, 0.1), bid(11, 100.0, 0.1), bid(12, 102.0, 0.1)],
        );
        // control: crossing displayed = 100(0.1)+101(0)+102(0.1)
        assert!(near(p, -0.2), "control got {p}");
    }
}

#[cfg(test)]
mod ab3465_review2_cases {
    use crate::{
        backtest::{
            Backtest, DataSource, ExchangeKind, L2AssetBuilder,
            assettype::LinearAsset,
            data::Data,
            models::{
                CommonFees, ConstantLatency, PowerProbQueueFunc3, ProbQueueModel,
                TradingValueFeeModel,
            },
        },
        depth::HashMapMarketDepth,
        prelude::Bot,
        types::*,
    };

    fn ev(t: i64, e: u64, px: f64, qty: f64) -> Event {
        Event { ev: e | EXCH_EVENT | LOCAL_EVENT, exch_ts: t, local_ts: t, px, qty, order_id: 0, ival: 0, fval: 0.0 }
    }
    fn bid(t: i64, px: f64, q: f64) -> Event { ev(t, DEPTH_EVENT | BUY_EVENT, px, q) }
    fn ask(t: i64, px: f64, q: f64) -> Event { ev(t, DEPTH_EVENT | SELL_EVENT, px, q) }
    fn buy_trade(t: i64, px: f64, q: f64) -> Event { ev(t, TRADE_EVENT | BUY_EVENT, px, q) }

    type Ord_ = (u64, bool, f64, f64, TimeInForce, i64); // id, buy, px, qty, tif, submit time

    fn run(pre: &[Event], orders: &[Ord_], later: &[Event], depletion: bool) -> (f64, usize) {
        let mut events = vec![bid(1, 98.0, 1.0), ask(1, 102.0, 1.0)];
        events.extend_from_slice(pre);
        events.push(ev(3, 0, 0.0, 0.0));
        events.extend_from_slice(later);
        events.push(ev(100, 0, 0.0, 0.0));
        events.sort_by_key(|e| e.exch_ts);
        let asset = L2AssetBuilder::default()
            .data(vec![DataSource::Data(Data::from_data(&events))])
            .latency_model(ConstantLatency::new(0, 0))
            .asset_type(LinearAsset::new(1.0))
            .fee_model(TradingValueFeeModel::new(CommonFees::new(0.00004, 0.00028)))
            .queue_model(ProbQueueModel::new(PowerProbQueueFunc3::new(3.0)))
            .exchange(ExchangeKind::PartialFillExchange)
            .own_fill_depletion(depletion)
            .depth(|| HashMapMarketDepth::new(1.0, 0.01))
            .build()
            .unwrap();
        let mut bt = Backtest::builder().add_asset(asset).build().unwrap();
        let mut now = 0;
        let mut os = orders.to_vec();
        os.sort_by_key(|o| o.5);
        for (id, b, px, q, tif, t) in os {
            if t > now { bt.elapse(t - now).unwrap(); now = t; }
            if b {
                bt.submit_buy_order(0, id, px, q, tif, OrdType::Limit, false).unwrap();
            } else {
                bt.submit_sell_order(0, id, px, q, tif, OrdType::Limit, false).unwrap();
            }
        }
        bt.elapse(100 - now).unwrap();
        let open = bt.orders(0).values().filter(|o| o.active()).count();
        (bt.position(0), open)
    }
    fn near(a: f64, b: f64) -> bool { (a - b).abs() < 1e-9 }

    // N1: two of our orders at the print price each get the whole remaining budget through the queue model.
    #[test]
    fn n1_equal_price_budget_shared() {
        let (p, _) = run(
            &[ask(2, 100.0, 0.3)],
            &[(1, false, 100.0, 1.0, TimeInForce::GTX, 3), (2, false, 100.0, 1.0, TimeInForce::GTX, 3)],
            &[buy_trade(10, 100.0, 0.8)],
            false,
        );
        assert!(near(p, -0.5), "0.8 print, 0.3 real ahead: 0.5 left for our two orders; got {p}");
    }
    // N1 control: one order through + one at print price share the budget (expected pass).
    #[test]
    fn n1b_through_then_equal() {
        let (p, _) = run(
            &[ask(2, 100.0, 0.3)],
            &[(1, false, 99.0, 0.4, TimeInForce::GTX, 3), (2, false, 100.0, 1.0, TimeInForce::GTX, 3)],
            &[buy_trade(10, 100.0, 0.8)],
            false,
        );
        assert!(near(p, -0.5), "got {p}");
    }

    // N2: a crossing level used before a depth clear and absent from the snapshot keeps its used qty forever.
    #[test]
    fn n2_clear_then_level_absent_then_new_liquidity() {
        let (p, _) = run(
            &[],
            &[(1, false, 100.0, 1.0, TimeInForce::GTX, 3)],
            &[
                bid(10, 101.0, 0.2),
                ev(11, DEPTH_CLEAR_EVENT | BUY_EVENT, 0.0, 0.0),
                ev(12, DEPTH_SNAPSHOT_EVENT | BUY_EVENT, 98.0, 1.0),
                bid(20, 101.0, 0.2),
            ],
            false,
        );
        assert!(near(p, -0.4), "new 0.2 at 101 after the old one vanished in the snapshot should fill; got {p}");
    }
    // N2 control: clear + snapshot re-sending the same level must not refill (expected pass).
    #[test]
    fn n2b_clear_resend_same() {
        let (p, _) = run(
            &[],
            &[(1, false, 100.0, 1.0, TimeInForce::GTX, 3)],
            &[
                bid(10, 101.0, 0.2),
                ev(11, DEPTH_CLEAR_EVENT | BUY_EVENT, 0.0, 0.0),
                ev(12, DEPTH_SNAPSHOT_EVENT | BUY_EVENT, 98.0, 1.0),
                ev(12, DEPTH_SNAPSHOT_EVENT | BUY_EVENT, 101.0, 0.2),
            ],
            false,
        );
        assert!(near(p, -0.2), "got {p}");
    }

    // N3: our crossed maker and our own IOC both take the same displayed crossing quantity.
    #[test]
    fn n3_ioc_and_crossed_maker_share_level() {
        for depletion in [true, false] {
            let (p, _) = run(
                &[],
                &[(1, true, 100.0, 1.0, TimeInForce::GTX, 3), (2, true, 99.0, 0.2, TimeInForce::IOC, 20)],
                &[ask(10, 99.0, 0.2)],
                depletion,
            );
            assert!(near(p, 0.2), "depletion={depletion}: 0.2 displayed at 99 fills 0.2 across our orders; got {p}");
        }
    }
    #[test]
    fn n3b_ioc_first_then_maker() {
        // IOC takes 0.2 at 99 at t=10 while our maker bid rests at 98.?; the maker is then crossed by an unrelated update.
        let (p, _) = run(
            &[ask(2, 99.0, 0.2)],
            &[(2, true, 99.0, 0.2, TimeInForce::IOC, 3), (1, true, 98.0, 1.0, TimeInForce::GTX, 3)],
            &[ask(10, 98.0, 0.1)],
            true,
        );
        // maker at 98 crossed by 98(0.1) only; 99 is above our price
        assert!(near(p, 0.3), "got {p}");
    }
}

#[cfg(test)]
mod ab3465_crossing_bench {
    use crate::{
        backtest::{
            Backtest, DataSource, ExchangeKind, L2AssetBuilder,
            assettype::LinearAsset,
            data::Data,
            models::{CommonFees, ConstantLatency, PowerProbQueueFunc3, ProbQueueModel, TradingValueFeeModel},
        },
        depth::HashMapMarketDepth,
        prelude::Bot,
        types::*,
    };
    fn ev(t: i64, e: u64, px: f64, qty: f64) -> Event {
        Event { ev: e | EXCH_EVENT | LOCAL_EVENT, exch_ts: t, local_ts: t, px, qty, order_id: 0, ival: 0, fval: 0.0 }
    }
    fn bench(n_distinct: u64, crossed: bool) {
        let m = 300_000i64;
        let mut events = vec![ev(1, DEPTH_EVENT | BUY_EVENT, 98.0, 1.0), ev(1, DEPTH_EVENT | SELL_EVENT, 102.0, 1.0), ev(3, 0, 0.0, 0.0), ev(4, 0, 0.0, 0.0), ev(5, 0, 0.0, 0.0)];
        if crossed {
            // our 1000.0 sell at 100 gets crossed by a bid far above, leaving a big remainder resting crossed
            events.push(ev(6, DEPTH_EVENT | BUY_EVENT, 100.0 + 500.0, 0.01));
        }
        for i in 0..m {
            events.push(ev(10 + i, DEPTH_EVENT | BUY_EVENT, 97.0 - (i % 50) as f64, 1.0 + (i % 7) as f64));
        }
        events.push(ev(10 + m + 1, 0, 0.0, 0.0));
        let asset = L2AssetBuilder::default()
            .data(vec![DataSource::Data(Data::from_data(&events))])
            .latency_model(ConstantLatency::new(0, 0))
            .asset_type(LinearAsset::new(1.0))
            .fee_model(TradingValueFeeModel::new(CommonFees::new(0.0, 0.0)))
            .queue_model(ProbQueueModel::new(PowerProbQueueFunc3::new(3.0)))
            .exchange(ExchangeKind::PartialFillExchange)
            .depth(|| HashMapMarketDepth::new(1.0, 0.01))
            .build()
            .unwrap();
        let mut bt = Backtest::builder().add_asset(asset).build().unwrap();
        bt.elapse(3).unwrap();
        for i in 0..n_distinct {
            bt.submit_sell_order(0, i + 10, 200.0 + i as f64, 0.01, TimeInForce::GTX, OrdType::Limit, false).unwrap();
        }
        if crossed {
            bt.submit_sell_order(0, 1, 100.0, 1000.0, TimeInForce::GTX, OrdType::Limit, false).unwrap();
        }
        bt.elapse(1).unwrap();
        for i in 0..n_distinct {
            bt.cancel(0, i + 10, false).unwrap();
        }
        bt.elapse(1).unwrap();
        let t0 = std::time::Instant::now();
        bt.elapse(m + 100).unwrap();
        println!("BENCH n_distinct={n_distinct} crossed={crossed}: {} bid events in {:?} pos={} ts={} open={}", m, t0.elapsed(), bt.position(0), bt.current_timestamp(), bt.orders(0).len());
    }
    #[test]
    #[ignore]
    fn bench_depth_events() {
        bench(0, false);
        bench(2000, false);
        bench(20000, false);
        bench(0, true);
    }
}

#[cfg(test)]
mod ab3465_review3_cases {
    use crate::{
        backtest::{
            Backtest, DataSource, ExchangeKind, L2AssetBuilder,
            assettype::LinearAsset,
            data::Data,
            models::{CommonFees, ConstantLatency, PowerProbQueueFunc3, ProbQueueModel, TradingValueFeeModel},
        },
        depth::HashMapMarketDepth,
        prelude::Bot,
        types::*,
    };

    fn ev(t: i64, e: u64, px: f64, qty: f64) -> Event {
        Event { ev: e | EXCH_EVENT | LOCAL_EVENT, exch_ts: t, local_ts: t, px, qty, order_id: 0, ival: 0, fval: 0.0 }
    }
    fn bid(t: i64, px: f64, q: f64) -> Event { ev(t, DEPTH_EVENT | BUY_EVENT, px, q) }
    fn ask(t: i64, px: f64, q: f64) -> Event { ev(t, DEPTH_EVENT | SELL_EVENT, px, q) }

    type O = (u64, bool, f64, f64, TimeInForce, i64);

    /// returns (position, open orders, num_trades)
    fn run(pre: &[Event], orders: &[O], later: &[Event], depletion: bool) -> (f64, usize, i64) {
        let mut events = vec![bid(1, 98.0, 1.0), ask(1, 102.0, 1.0)];
        events.extend_from_slice(pre);
        events.push(ev(3, 0, 0.0, 0.0));
        events.extend_from_slice(later);
        events.push(ev(100, 0, 0.0, 0.0));
        events.sort_by_key(|e| e.exch_ts);
        let asset = L2AssetBuilder::default()
            .data(vec![DataSource::Data(Data::from_data(&events))])
            .latency_model(ConstantLatency::new(0, 0))
            .asset_type(LinearAsset::new(1.0))
            .fee_model(TradingValueFeeModel::new(CommonFees::new(0.00004, 0.00028)))
            .queue_model(ProbQueueModel::new(PowerProbQueueFunc3::new(3.0)))
            .exchange(ExchangeKind::PartialFillExchange)
            .own_fill_depletion(depletion)
            .depth(|| HashMapMarketDepth::new(1.0, 0.01))
            .build()
            .unwrap();
        let mut bt = Backtest::builder().add_asset(asset).build().unwrap();
        let mut now = 0;
        let mut os = orders.to_vec();
        os.sort_by_key(|o| o.5);
        for (id, b, px, q, tif, t) in os {
            if t > now { bt.elapse(t - now).unwrap(); now = t; }
            if b {
                bt.submit_buy_order(0, id, px, q, tif, OrdType::Limit, false).unwrap();
            } else {
                bt.submit_sell_order(0, id, px, q, tif, OrdType::Limit, false).unwrap();
            }
        }
        bt.elapse(100 - now).unwrap();
        let open = bt.orders(0).values().filter(|o| o.active()).count();
        for o in bt.orders(0).values() { println!("DBG id={} st={:?} leaves={:e} exec={:e} maker={}", o.order_id, o.status, o.leaves_qty, o.exec_qty, o.maker); }
        println!("DBG pos={:e} n={}", bt.position(0), bt.state_values(0).num_trades);
        (bt.position(0), open, bt.state_values(0).num_trades)
    }
    fn near(a: f64, b: f64) -> bool { (a - b).abs() < 1e-9 }

    // F1: the crossing walk stops at our LEAST crossed order, so deeper levels that cross only our more aggressive
    // order are never visited.
    #[test]
    fn f1_walk_reaches_the_most_crossed_order() {
        let (p, _, _) = run(
            &[],
            &[(1, false, 99.0, 1.0, TimeInForce::GTX, 3), (2, false, 100.0, 1.0, TimeInForce::GTX, 3)],
            &[bid(10, 101.0, 0.2), bid(11, 99.0, 0.5)],
            false,
        );
        // 101(0.2) fills our 99 first; the new bid at 99 (0.5) crosses our 99 ask for 0.5 more
        assert!(near(p, -0.7), "got {p}");
    }
    #[test]
    fn f1b_buy_mirror() {
        let (p, _, _) = run(
            &[],
            &[(1, true, 101.0, 1.0, TimeInForce::GTX, 3), (2, true, 100.0, 1.0, TimeInForce::GTX, 3)],
            &[ask(10, 99.0, 0.2), ask(11, 101.0, 0.5)],
            false,
        );
        assert!(near(p, 0.7), "got {p}");
    }

    // F2: IOC walks displayed - crossed with no lot guard: float residue becomes a dust fill.
    #[test]
    fn f2_ioc_dust_fill_from_crossed_residue() {
        let (p, _, n) = run(
            &[],
            &[(1, true, 100.0, 1.0, TimeInForce::GTX, 3), (2, true, 99.0, 0.5, TimeInForce::IOC, 20)],
            &[ask(10, 99.0, 0.05), ask(11, 99.0, 0.21)],
            false,
        );
        // maker takes 0.05 then 0.16 = all 0.21 displayed; the IOC finds nothing
        assert!(near(p, 0.21) && n == 2, "pos={p} num_trades={n} (expected 2 maker fills, no IOC fill)");
    }

    // F3: Lighter live-book style resync: EXCH_DEPTH_CLEAR then plain depth events at the same ts
    // (asks first, then bids ascending), used level re-sent must not refill; later ticker at later ts.
    #[test]
    fn f3_lighter_style_resync_no_refill() {
        let (p, _, _) = run(
            &[],
            &[(1, false, 100.0, 1.0, TimeInForce::GTX, 3)],
            &[
                bid(10, 101.0, 0.2),
                ev(20, DEPTH_CLEAR_EVENT, 0.0, 0.0),
                ask(20, 102.0, 1.0),
                bid(20, 98.0, 1.0),
                bid(20, 101.0, 0.2),
                bid(30, 97.0, 1.0),
            ],
            false,
        );
        assert!(near(p, -0.2), "got {p}");
    }
    #[test]
    fn f3b_resync_absent_then_new() {
        let (p, _, _) = run(
            &[],
            &[(1, false, 100.0, 1.0, TimeInForce::GTX, 3)],
            &[
                bid(10, 101.0, 0.2),
                ev(20, DEPTH_CLEAR_EVENT, 0.0, 0.0),
                ask(20, 102.0, 1.0),
                bid(20, 98.0, 1.0),
                bid(30, 97.0, 1.0),
                bid(40, 101.0, 0.2),
            ],
            false,
        );
        assert!(near(p, -0.4), "got {p}");
    }
    // F3c: resync re-sends the level at the same ts but the first event after the clear is on the other side only.
    #[test]
    fn f3c_resync_then_immediate_change_same_ts() {
        // A real delta at the resync ts that removes 101 and re-adds it at a later ts: must fill again.
        let (p, _, _) = run(
            &[],
            &[(1, false, 100.0, 1.0, TimeInForce::GTX, 3)],
            &[
                bid(10, 101.0, 0.2),
                ev(20, DEPTH_CLEAR_EVENT, 0.0, 0.0),
                bid(20, 98.0, 1.0),
                bid(25, 101.0, 0.2),
            ],
            false,
        );
        // 101 was absent from the resync; a new 0.2 at t=25 is new liquidity
        assert!(near(p, -0.4), "got {p}");
    }

    // F4: modify (cancel+new at a new price) and partial fill/cancel keep the ladders consistent: no panic.
    #[test]
    fn f4_cancel_after_partial_and_trade() {
        let (p, open, _) = run(
            &[],
            &[(1, false, 100.0, 1.0, TimeInForce::GTX, 3), (2, false, 100.0, 1.0, TimeInForce::GTX, 3)],
            &[bid(10, 101.0, 0.5), bid(11, 101.0, 1.8), ev(12, TRADE_EVENT | BUY_EVENT, 103.0, 0.5)],
            false,
        );
        assert!(near(p, -2.0) && open == 0, "got {p} open={open}");
    }
}

/// AB#3465 (gentilly): a print and the depth update that publishes its depletion are de-duplicated by the exchange in
/// either feed order when the depth event carries its publishing batch's stamp (`ival`, the batch's send time).
#[cfg(test)]
mod ab3465_batch_depletion_exchange {
    use crate::{
        backtest::{
            Backtest, DataSource, ExchangeKind, L2AssetBuilder,
            assettype::LinearAsset,
            data::Data,
            models::{
                CommonFees, ConstantLatency, PowerProbQueueFunc3, ProbQueueModel,
                TradingValueFeeModel,
            },
        },
        depth::HashMapMarketDepth,
        prelude::Bot,
        types::*,
    };

    fn event(t: i64, ev: u64, px: f64, qty: f64, stamp: i64) -> Event {
        Event {
            ev: ev | EXCH_EVENT | LOCAL_EVENT,
            exch_ts: t,
            local_ts: t,
            px,
            qty,
            order_id: 0,
            ival: stamp,
            fval: 0.0,
        }
    }

    /// 0.001 rests at the ask 100; our 0.001 sell joins behind it at t=3; then a 0.0006 buy print at t=40 and the
    /// batch stamped 50 publishing the level at 0.0004 (its row at pseudo time `depletion_at`).
    fn position(depletion_at: i64, stamp: i64) -> f64 {
        let mut events = vec![
            event(1, DEPTH_EVENT | BUY_EVENT, 98.0, 1.0, 0),
            event(2, DEPTH_EVENT | SELL_EVENT, 100.0, 0.001, 0),
            event(3, 0, 0.0, 0.0, 0),
        ];
        let mut later = vec![
            event(40, TRADE_EVENT | BUY_EVENT, 100.0, 0.0006, 0),
            event(depletion_at, DEPTH_EVENT | SELL_EVENT, 100.0, 0.0004, stamp),
        ];
        later.sort_by_key(|e| e.exch_ts);
        events.extend(later);
        events.push(event(60, 0, 0.0, 0.0, 0));
        let asset = L2AssetBuilder::default()
            .data(vec![DataSource::Data(Data::from_data(&events))])
            .latency_model(ConstantLatency::new(0, 0))
            .asset_type(LinearAsset::new(1.0))
            .fee_model(TradingValueFeeModel::new(CommonFees::new(0.00004, 0.00028)))
            .queue_model(ProbQueueModel::new(PowerProbQueueFunc3::new(0.5)))
            .exchange(ExchangeKind::PartialFillExchange)
            .depth(|| HashMapMarketDepth::new(1.0, 0.00001))
            .build()
            .unwrap();
        let mut bt = Backtest::builder().add_asset(asset).build().unwrap();
        bt.elapse(3).unwrap();
        bt.submit_sell_order(0, 1, 100.0, 0.001, TimeInForce::GTX, OrdType::Limit, false)
            .unwrap();
        bt.elapse(60).unwrap();
        bt.position(0)
    }

    #[test]
    fn a_print_that_does_not_reach_our_order_never_fills_it_in_either_order() {
        assert_eq!(position(45, 50), 0.0, "print first");
        assert_eq!(
            position(35, 50),
            0.0,
            "depletion of the print's own batch first"
        );
    }

    #[test]
    fn an_unstamped_feed_keeps_the_previous_behaviour() {
        assert_eq!(position(45, 0), 0.0, "print first");
        assert!(
            position(35, 0) < 0.0,
            "unstamped depletion first: the old double count"
        );
    }

    #[test]
    fn a_depletion_of_an_earlier_batch_does_not_explain_a_later_print() {
        // The decrease belongs to a batch stamped 38, before the print at 40: it was cancels, the print is new.
        assert!(position(35, 38) < 0.0);
    }
}
