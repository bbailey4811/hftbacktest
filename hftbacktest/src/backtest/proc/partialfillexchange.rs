use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    rc::Rc,
};

use crate::{
    backtest::{
        BacktestError,
        assettype::AssetType,
        models::{FeeModel, LatencyModel, QueueModel},
        order::ExchToLocal,
        proc::Processor,
        state::State,
    },
    depth::{INVALID_MAX, INVALID_MIN, L2MarketDepth, MarketDepth},
    prelude::OrdType,
    types::{
        EXCH_ASK_DEPTH_CLEAR_EVENT, EXCH_ASK_DEPTH_EVENT, EXCH_ASK_DEPTH_SNAPSHOT_EVENT,
        EXCH_BID_DEPTH_CLEAR_EVENT, EXCH_BID_DEPTH_EVENT, EXCH_BID_DEPTH_SNAPSHOT_EVENT,
        EXCH_BUY_TRADE_EVENT, EXCH_DEPTH_CLEAR_EVENT, EXCH_EVENT, EXCH_SELL_TRADE_EVENT, Event,
        Order, OrderId, Side, Status, TimeInForce,
    },
};

/// AB#3465 (gentilly): a print's own execution time, which decides whether it belongs to a stamped depth batch. A
/// feed that matches prints at another time (a live simulator applying frames at receipt) states the print's own time
/// in `ival`; otherwise the matching time is the print's own time.
fn trade_time(event: &Event) -> i64 {
    if event.ival > 0 { event.ival } else { event.exch_ts }
}

/// AB#3465 (gentilly): how many ticks past the opposite best a crossed resting order is matched against.
const MAX_CROSS_LEVELS: i64 = 1_000;

/// Applies a feed update of one price level to what our own taker fills took from it. `level_qty` is the quantity
/// the depth holds at the level after the update (zero once the depth has dropped the level). Without `persists`
/// the feed update makes the level whole again. With it, what we took stays taken, capped at the quantity the level
/// now holds, until the level is empty.
fn own_fills_after_level_update(
    consumed: &mut HashMap<i64, f64>,
    persists: bool,
    price_tick: i64,
    level_qty: f64,
) {
    if !persists || level_qty <= 0.0 {
        consumed.remove(&price_tick);
    } else if let Some(taken) = consumed.get_mut(&price_tick) {
        if *taken > level_qty {
            *taken = level_qty;
        }
    }
}

/// The quantity of a level a taker order can still fill against, given what our own fills took from it. With
/// persisting depletion a level can stay fully taken across many orders, so a remainder below half a lot, which is
/// only the rounding left by adding our fills up, is nothing rather than a fill of that size.
fn own_fill_available(level_qty: f64, taken: f64, persists: bool, lot_size: f64) -> f64 {
    let available = level_qty - taken;
    if persists && available < 0.5 * lot_size {
        0.0
    } else {
        available
    }
}

/// The exchange model with partial fills.
///
/// * Support order types: [OrdType::Limit](crate::types::OrdType::Limit)
/// * Support time-in-force: [`TimeInForce::GTC`], [`TimeInForce::FOK`], [`TimeInForce::IOC`],
///   [`TimeInForce::GTX`]
///
/// **Execution of resting orders (AB#3465, gentilly)**
/// A resting order is never filled in full by one event; it is filled only for the quantity that would have met it.
///
/// - A trade printed through our price: the taker would have met our orders first, best price first, for the
///   quantity it traded. One print is one budget shared by all our orders on that side, and what is left of it
///   reaches orders at the print's price through the queue model.
/// - The opposite book displayed at or through our price: each crossing level's quantity would have met our
///   orders, best price first, once. What our orders took from a level stays taken across its feed updates
///   (capped at what it displays, forgotten when it empties), so only quantity added later fills again. The check
///   runs on every update of the opposite side, so a remainder resting behind the opposite best keeps filling.
///
/// **Conditions for Partial Execution**
/// Buy order in the order book
///
/// - Filled by (remaining) sell trade quantity: your order is at the front of the queue && your
///   order price == sell trade price
///
/// Sell order in the order book
///
/// - Filled by (remaining) buy trade quantity: your order is at the front of the queue && your
///   order price == buy trade price
///
/// **Liquidity-Taking Order**
/// Liquidity-taking orders will be executed based on the quantity of the order book, even though
/// the best price and quantity do not change due to your execution. Be aware that this may cause
/// unrealistic fill simulations if you attempt to execute a large quantity.
///
/// **General Comment**
/// Simulating partial fills accurately can be challenging, as they may indicate potential market
/// impact. The rule of thumb is to ensure that your backtesting results align with your live
/// results.
/// (more comment will be added...)
///
pub struct PartialFillExchange<AT, LM, QM, MD, FM>
where
    AT: AssetType,
    LM: LatencyModel,
    QM: QueueModel<MD>,
    MD: MarketDepth,
    FM: FeeModel,
{
    // key: order_id, value: Order
    orders: Rc<RefCell<HashMap<OrderId, Order>>>,
    // key: order's price tick, value: order_ids
    buy_orders: HashMap<i64, HashSet<OrderId>>,
    sell_orders: HashMap<i64, HashSet<OrderId>>,

    order_e2l: ExchToLocal<LM>,

    depth: MD,
    state: State<AT, FM>,
    queue_model: QM,

    filled_orders: Vec<OrderId>,

    // Own-fill depletion (opt-in): the quantity our own IOC taker fills took at each price level, since that level's
    // last feed update or, with `own_fill_depletion_persists`, since the level was last empty. Replayed depth never
    // reflects our own fills, so without it a later taker order can fill again against displayed shares one of
    // ours already took. A depth clear event forgets the whole side.
    own_fill_depletion: bool,
    // With own-fill depletion: `false` (the default) forgets what we took from a level when the feed next updates
    // that level; `true` keeps it across feed updates until the feed empties the level.
    own_fill_depletion_persists: bool,
    consumed_bid: HashMap<i64, f64>,
    consumed_ask: HashMap<i64, f64>,
    // AB#3465 (gentilly): per price level of the opposite book, the quantity it displayed at or through our resting
    // orders' prices that has already filled them. Kept across that level's feed updates, capped at what it
    // displays and forgotten when it empties, so the same displayed quantity fills our orders only once.
    crossed_bid: HashMap<i64, f64>,
    crossed_ask: HashMap<i64, f64>,
    // A depth clear keeps those, so the rebuild re-sending a level does not fill us again. The rebuild is the
    // snapshot events and the depth events stamped like the first one after the clear; the first later depth event
    // of that side drops the levels the rebuilt book no longer displays. `Some(None)`: cleared, rebuild not begun.
    bid_rebuild: Option<Option<i64>>,
    ask_rebuild: Option<Option<i64>>,
    // AB#3465 (gentilly): per price level, the publishing batch (its send time, from a depth event's `ival`) of the
    // level's latest stamped decrease and how much of that decrease no print of the batch has explained yet. A print
    // of that batch seen after its depletion is de-duplicated against it (see `QueueModel::trade_in_batch`). Feeds
    // that stamp no batch (`ival` 0) never fill these, and keep the behaviour from before.
    batch_decrease_bid: HashMap<i64, (i64, f64)>,
    batch_decrease_ask: HashMap<i64, (i64, f64)>,
    // AB#3465 (gentilly): when each resting order arrived at the exchange at its current price (a re-price arrives
    // anew; a size cut keeps its time), and, while an event is processed as of an earlier time, that time: only the
    // orders that stood by then are met by its trade or crossing.
    arrived: HashMap<OrderId, i64>,
    as_of: Option<i64>,
    // AB#3465 (gentilly): per resting post-only order not yet settled, the latest book recorded before it arrived
    // (as of when, and whether it crossed the order).
    post_only_verdicts: HashMap<OrderId, (i64, bool)>,
}

impl<AT, LM, QM, MD, FM> PartialFillExchange<AT, LM, QM, MD, FM>
where
    AT: AssetType,
    LM: LatencyModel,
    QM: QueueModel<MD>,
    MD: MarketDepth,
    FM: FeeModel,
{
    /// Constructs an instance of `PartialFillExchange`.
    pub fn new(
        depth: MD,
        state: State<AT, FM>,
        queue_model: QM,
        order_e2l: ExchToLocal<LM>,
    ) -> Self {
        Self {
            orders: Default::default(),
            buy_orders: Default::default(),
            sell_orders: Default::default(),
            order_e2l,
            depth,
            state,
            queue_model,
            filled_orders: Default::default(),
            own_fill_depletion: false,
            own_fill_depletion_persists: false,
            consumed_bid: Default::default(),
            consumed_ask: Default::default(),
            crossed_bid: Default::default(),
            crossed_ask: Default::default(),
            bid_rebuild: None,
            ask_rebuild: None,
            batch_decrease_bid: Default::default(),
            batch_decrease_ask: Default::default(),
            arrived: Default::default(),
            as_of: None,
            post_only_verdicts: Default::default(),
        }
    }

    /// Opt-in own-fill depletion: an IOC taker order cannot fill against displayed quantity that our own earlier IOC
    /// taker fills already took at that price level, until the feed next updates the level (or, with
    /// [`Self::with_own_fill_depletion_persists`], until the level is empty).
    pub fn with_own_fill_depletion(mut self, enabled: bool) -> Self {
        self.own_fill_depletion = enabled;
        self
    }

    /// With own-fill depletion enabled, what our own IOC taker fills took from a price level stays taken across feed
    /// updates of that level instead of being forgotten at the next one. An update that leaves less than we took
    /// caps what we took at the new displayed quantity, quantity displayed beyond what we took is available, and a
    /// level the feed empties starts fresh. Across level updates this is the conservative bound: it assumes an
    /// update never replaces what we took, where the default assumes it always does. A depth clear event still
    /// forgets the whole side, and only IOC orders read or add to what was taken.
    pub fn with_own_fill_depletion_persists(mut self, enabled: bool) -> Self {
        self.own_fill_depletion_persists = enabled;
        self
    }

    /// AB#3465 (gentilly): our resting orders on `side` whose price a trade at `price_tick` met, best price first:
    /// strictly through the print for a sell order `price < print`, at the print for `price == print`.
    fn orders_met_by_trade(&self, side: Side, price_tick: i64) -> Vec<(i64, OrderId)> {
        let book = if side == Side::Sell {
            &self.sell_orders
        } else {
            &self.buy_orders
        };
        let mut met: Vec<(i64, OrderId)> = book
            .iter()
            .filter(|(tick, _)| {
                if side == Side::Sell {
                    **tick <= price_tick
                } else {
                    **tick >= price_tick
                }
            })
            .flat_map(|(tick, ids)| ids.iter().map(move |id| (*tick, *id)))
            // AB#3465 (gentilly): processed as of an earlier time, only the orders that stood by then.
            .filter(|(_, id)| {
                self.as_of
                    .is_none_or(|as_of| self.arrived.get(id).is_some_and(|at| *at <= as_of))
            })
            .collect();
        if side == Side::Sell {
            met.sort_unstable();
        } else {
            met.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        }
        met
    }

    /// A trade of `qty` at `price_tick` against our resting orders on `side`. Orders priced through the print fill
    /// from it best price first; what is left of it reaches orders at the print's price through the queue model.
    fn match_trade(
        &mut self,
        side: Side,
        price_tick: i64,
        qty: f64,
        timestamp: i64,
        trade_time: i64,
    ) -> Result<(), BacktestError> {
        let mut budget = qty;
        // AB#3465: the part of this print its own publishing batch already took off the level before the print.
        let level_decreases = if side == Side::Sell {
            &mut self.batch_decrease_ask
        } else {
            &mut self.batch_decrease_bid
        };
        let explained = match level_decreases.get_mut(&price_tick) {
            Some((stamp, unexplained)) if trade_time <= *stamp && *unexplained > 0.0 => {
                let explained = qty.min(*unexplained);
                *unexplained -= explained;
                explained
            }
            _ => 0.0,
        };
        let orders = self.orders.clone();
        let mut orders_borrowed = orders.borrow_mut();
        for (tick, order_id) in self.orders_met_by_trade(side, price_tick) {
            let order = orders_borrowed.get_mut(&order_id).unwrap();
            if tick != price_tick {
                let exec_qty = budget.min(order.leaves_qty);
                if (exec_qty / self.depth.lot_size()).round() <= 0.0 {
                    continue;
                }
                budget -= exec_qty;
                self.fill::<true>(order, timestamp, true, order.price_tick, exec_qty)?;
            } else {
                if budget <= 0.0 {
                    break;
                }
                // Updates the order's queue position.
                self.queue_model.trade_in_batch(
                    order,
                    budget,
                    &self.depth,
                    timestamp,
                    explained.min(budget),
                );
                let filled_qty = self.queue_model.is_filled(order, &self.depth);
                if filled_qty > 0.0 {
                    // q_ahead is negative since is_filled is true and its value represents the
                    // executable quantity of this order after execution in the queue ahead of this
                    // order.
                    let exec_qty = filled_qty.min(order.leaves_qty);
                    budget -= exec_qty;
                    self.fill::<true>(order, timestamp, true, order.price_tick, exec_qty)?;
                }
            }
            // The fill owns lot-rounding and terminal status. Equality (and a
            // sub-half-lot remainder) also exhausts the resting order.
            if order.status == Status::Filled {
                self.filled_orders.push(order_id);
            }
        }
        drop(orders_borrowed);
        self.remove_filled_orders();
        Ok(())
    }

    /// AB#3465 (gentilly): while an event is processed as of an earlier time, the book it leaves stood on the
    /// exchange from then. For each resting post-only (GTX) order that arrived after that time, whether that book
    /// crosses it (its price at or through the opposite best) is its verdict as of then. A later such event (as of a
    /// time still before the order arrived) overwrites it; an earlier one does not. The book the order met on arrival
    /// is the one the latest of them left ([`Self::settle_post_only_arrivals`]).
    fn record_post_only_verdicts(&mut self, as_of: i64) {
        let best_ask = self.depth.best_ask_tick();
        let best_bid = self.depth.best_bid_tick();
        let orders = self.orders.borrow();
        for (side, ladder) in [
            (Side::Buy, &self.buy_orders),
            (Side::Sell, &self.sell_orders),
        ] {
            for (tick, ids) in ladder {
                let crossed = if side == Side::Buy {
                    best_ask != INVALID_MAX && *tick >= best_ask
                } else {
                    best_bid != INVALID_MIN && *tick <= best_bid
                };
                for id in ids {
                    if !self.arrived.get(id).is_some_and(|at| *at > as_of)
                        || orders[id].time_in_force != TimeInForce::GTX
                    {
                        continue;
                    }
                    let verdict = self
                        .post_only_verdicts
                        .entry(*id)
                        .or_insert((as_of, crossed));
                    if verdict.0 <= as_of {
                        *verdict = (as_of, crossed);
                    }
                }
            }
        }
    }

    /// AB#3465 (gentilly): before an event as of `horizon` meets any order (a trade or a crossing), each resting
    /// post-only order that arrived by `horizon` has met its book: if the latest book recorded before its arrival
    /// ([`Self::record_post_only_verdicts`]) crossed it, the exchange rejected it on arrival, and it is expired now,
    /// when that book is first known, never filling from it. An order no earlier book was recorded for met the book
    /// it was accepted on.
    fn settle_post_only_arrivals(&mut self, horizon: i64, timestamp: i64) {
        if self.post_only_verdicts.is_empty() {
            return;
        }
        let settled: Vec<(OrderId, bool)> = self
            .post_only_verdicts
            .iter()
            .filter(|(id, _)| self.arrived.get(id).is_none_or(|at| *at <= horizon))
            .map(|(id, (_, crossed))| (*id, *crossed))
            .collect();
        for (order_id, crossed) in settled {
            self.post_only_verdicts.remove(&order_id);
            if !crossed || !self.arrived.contains_key(&order_id) {
                continue;
            }
            let mut order = self
                .orders
                .borrow_mut()
                .remove(&order_id)
                .expect("AB#3465 an order with an arrival time is resting");
            self.arrived.remove(&order_id);
            self.unladder(order.side, order.price_tick, order_id);
            order.status = Status::Expired;
            order.exch_timestamp = timestamp;
            self.order_e2l.respond(order);
        }
    }

    /// AB#3465 (gentilly): the opposite book displays quantity at or through the price of our resting orders on
    /// `side`. Each crossing level fills them, best price first, from the quantity it displays that has not already
    /// filled them, visiting levels from the most aggressive one down.
    fn match_crossing(&mut self, side: Side, timestamp: i64) -> Result<(), BacktestError> {
        let best = if side == Side::Sell {
            self.depth.best_bid_tick()
        } else {
            self.depth.best_ask_tick()
        };
        if best == INVALID_MIN || best == INVALID_MAX {
            return Ok(());
        }
        // Our orders the opposite best has reached, best price first.
        let crossed = self.orders_met_by_trade(side, best);
        // Best price first: the first is our most crossed order.
        let Some(&(deepest, _)) = crossed.first() else {
            return Ok(());
        };
        // Crossing levels, most aggressive first, from the opposite best to our deepest crossed order. The walk is
        // bounded: a remainder crossed further than MAX_CROSS_LEVELS ticks fills only from the nearest levels.
        let span = (best - deepest).abs().min(MAX_CROSS_LEVELS);
        let orders = self.orders.clone();
        for step in 0..=span {
            let level = if side == Side::Sell {
                best - step
            } else {
                best + step
            };
            let displayed = if side == Side::Sell {
                self.depth.bid_qty_at_tick(level)
            } else {
                self.depth.ask_qty_at_tick(level)
            };
            if displayed <= 0.0 {
                continue;
            }
            let used_map = if side == Side::Sell {
                &self.crossed_bid
            } else {
                &self.crossed_ask
            };
            let mut available = displayed - used_map.get(&level).copied().unwrap_or(0.0);
            if self.own_fill_depletion {
                // Displayed quantity our own IOC orders already took cannot meet our resting orders too.
                let consumed = if side == Side::Sell {
                    &self.consumed_bid
                } else {
                    &self.consumed_ask
                };
                available -= consumed.get(&level).copied().unwrap_or(0.0);
            }
            let mut used_now = 0.0;
            // A bid at `level` meets our sells priced at or below it; an ask meets our buys at or above it.
            let mut orders_borrowed = orders.borrow_mut();
            for &(tick, order_id) in &crossed {
                let meets = if side == Side::Sell {
                    tick <= level
                } else {
                    tick >= level
                };
                let Some(order) = orders_borrowed.get_mut(&order_id) else {
                    continue;
                };
                if !meets || order.status == Status::Filled {
                    continue;
                }
                let exec_qty = available.min(order.leaves_qty);
                if (exec_qty / self.depth.lot_size()).round() <= 0.0 {
                    break;
                }
                available -= exec_qty;
                used_now += exec_qty;
                self.fill::<true>(order, timestamp, true, order.price_tick, exec_qty)?;
                if order.status == Status::Filled {
                    self.filled_orders.push(order_id);
                }
            }
            if used_now > 0.0 {
                let used_map = if side == Side::Sell {
                    &mut self.crossed_bid
                } else {
                    &mut self.crossed_ask
                };
                *used_map.entry(level).or_insert(0.0) += used_now;
            }
        }
        self.remove_filled_orders();
        Ok(())
    }

    fn fill<const MAKE_RESPONSE: bool>(
        &mut self,
        order: &mut Order,
        timestamp: i64,
        maker: bool,
        exec_price_tick: i64,
        exec_qty: f64,
    ) -> Result<(), BacktestError> {
        if order.status == Status::Expired
            || order.status == Status::Canceled
            || order.status == Status::Filled
        {
            return Err(BacktestError::InvalidOrderStatus);
        }

        order.maker = maker;
        if maker {
            order.exec_price_tick = order.price_tick;
        } else {
            order.exec_price_tick = exec_price_tick;
        }

        order.exec_qty = exec_qty;
        order.leaves_qty -= exec_qty;
        if (order.leaves_qty / self.depth.lot_size()).round() > 0f64 {
            order.status = Status::PartiallyFilled;
        } else {
            order.status = Status::Filled;
        }
        order.exch_timestamp = timestamp;

        self.state.apply_fill(order);

        if MAKE_RESPONSE {
            self.order_e2l.respond(order.clone());
        }
        Ok(())
    }

    /// Removes an order id from its side's price ladder, and the price from the ladder once no order rests there
    /// (AB#3465: the ladders are scanned on every event, so they must hold only prices with resting orders).
    fn unladder(&mut self, side: Side, price_tick: i64, order_id: OrderId) {
        let ladder = if side == Side::Buy {
            &mut self.buy_orders
        } else {
            &mut self.sell_orders
        };
        let ids = ladder.get_mut(&price_tick).unwrap();
        ids.remove(&order_id);
        if ids.is_empty() {
            ladder.remove(&price_tick);
        }
    }

    fn remove_filled_orders(&mut self) {
        if !self.filled_orders.is_empty() {
            let orders = self.orders.clone();
            let mut orders = orders.borrow_mut();
            for order_id in std::mem::take(&mut self.filled_orders) {
                let order = orders.remove(&order_id).unwrap();
                self.arrived.remove(&order_id);
                self.unladder(order.side, order.price_tick, order_id);
            }
        }
    }

    fn on_bid_qty_chg(&mut self, price_tick: i64, prev_qty: f64, new_qty: f64, batch_stamp: i64) {
        // AB#3465: the batch's depletion at this level before this update that no print has explained yet (an
        // order joining between two of the batch's updates never saw it).
        let unseen = match self.batch_decrease_bid.get(&price_tick) {
            Some(&(stamp, unexplained)) if batch_stamp > 0 && stamp == batch_stamp => unexplained,
            _ => 0.0,
        };
        // AB#3465: a stamped decrease is remembered per level until a print of its batch explains it.
        if batch_stamp > 0 {
            if new_qty < prev_qty {
                let entry = self
                    .batch_decrease_bid
                    .entry(price_tick)
                    .or_insert((batch_stamp, 0.0));
                if entry.0 != batch_stamp {
                    *entry = (batch_stamp, 0.0);
                }
                entry.1 += prev_qty - new_qty;
            }
        }
        let orders = self.orders.clone();
        if let Some(order_ids) = self.buy_orders.get(&price_tick) {
            for order_id in order_ids.iter() {
                let mut orders_borrowed = orders.borrow_mut();
                let order = orders_borrowed.get_mut(order_id).unwrap();
                if batch_stamp > 0 {
                    self.queue_model.depth_in_batch(
                        order,
                        prev_qty,
                        new_qty,
                        &self.depth,
                        batch_stamp,
                        unseen,
                    );
                } else {
                    self.queue_model
                        .depth(order, prev_qty, new_qty, &self.depth);
                }
            }
        }
    }

    fn on_ask_qty_chg(&mut self, price_tick: i64, prev_qty: f64, new_qty: f64, batch_stamp: i64) {
        // AB#3465: the batch's depletion at this level before this update that no print has explained yet (an
        // order joining between two of the batch's updates never saw it).
        let unseen = match self.batch_decrease_ask.get(&price_tick) {
            Some(&(stamp, unexplained)) if batch_stamp > 0 && stamp == batch_stamp => unexplained,
            _ => 0.0,
        };
        // AB#3465: a stamped decrease is remembered per level until a print of its batch explains it.
        if batch_stamp > 0 {
            if new_qty < prev_qty {
                let entry = self
                    .batch_decrease_ask
                    .entry(price_tick)
                    .or_insert((batch_stamp, 0.0));
                if entry.0 != batch_stamp {
                    *entry = (batch_stamp, 0.0);
                }
                entry.1 += prev_qty - new_qty;
            }
        }
        let orders = self.orders.clone();
        if let Some(order_ids) = self.sell_orders.get(&price_tick) {
            for order_id in order_ids.iter() {
                let mut orders_borrowed = orders.borrow_mut();
                let order = orders_borrowed.get_mut(order_id).unwrap();
                if batch_stamp > 0 {
                    self.queue_model.depth_in_batch(
                        order,
                        prev_qty,
                        new_qty,
                        &self.depth,
                        batch_stamp,
                        unseen,
                    );
                } else {
                    self.queue_model
                        .depth(order, prev_qty, new_qty, &self.depth);
                }
            }
        }
    }

    // AB#3112 (gentilly): every level a taker walks is reported in its own
    // response (`fill::<true>`), so the local processor can book each fill at
    // its own price; the final ack response then reports no new execution.
    fn ack_new(&mut self, order: &mut Order, timestamp: i64) -> Result<(), BacktestError> {
        let acked = self.ack_new_order(order, timestamp);
        // AB#3465 (gentilly): a resting order stands on the exchange from the time it arrived.
        if self.orders.borrow().contains_key(&order.order_id) {
            self.arrived.insert(order.order_id, timestamp);
        }
        acked
    }

    fn ack_new_order(&mut self, order: &mut Order, timestamp: i64) -> Result<(), BacktestError> {
        if self.orders.borrow().contains_key(&order.order_id) {
            return Err(BacktestError::OrderIdExist);
        }

        if order.side == Side::Buy {
            match order.order_type {
                OrdType::Limit => {
                    // Checks if the buy order price is greater than or equal to the current best ask.
                    if order.price_tick >= self.depth.best_ask_tick() {
                        match order.time_in_force {
                            TimeInForce::GTX => {
                                order.status = Status::Expired;
                                order.exch_timestamp = timestamp;
                                Ok(())
                            }
                            TimeInForce::FOK => {
                                // The order must be executed immediately in its entirety; otherwise, the
                                // entire order will be cancelled.
                                let mut execute = false;
                                let mut cum_qty = 0f64;
                                for t in self.depth.best_ask_tick()..=order.price_tick {
                                    cum_qty += self.depth.ask_qty_at_tick(t);
                                    if (cum_qty / self.depth.lot_size()).round()
                                        >= (order.qty / self.depth.lot_size()).round()
                                    {
                                        execute = true;
                                        break;
                                    }
                                }
                                if execute {
                                    for t in self.depth.best_ask_tick()..=order.price_tick {
                                        let qty = self.depth.ask_qty_at_tick(t);
                                        if qty > 0.0 {
                                            let exec_qty = qty.min(order.leaves_qty);
                                            self.fill::<true>(
                                                order, timestamp, false, t, exec_qty,
                                            )?;
                                            if order.status == Status::Filled {
                                                return Ok(());
                                            }
                                        }
                                    }
                                    unreachable!();
                                } else {
                                    order.status = Status::Expired;
                                    order.exch_timestamp = timestamp;
                                    Ok(())
                                }
                            }
                            TimeInForce::IOC => {
                                // The order must be executed immediately.
                                for t in self.depth.best_ask_tick()..=order.price_tick {
                                    // AB#3465 (gentilly): displayed quantity our crossed resting orders already
                                    // took is gone for our IOC orders too.
                                    let mut qty = self.depth.ask_qty_at_tick(t)
                                        - self.crossed_ask.get(&t).copied().unwrap_or(0.0);
                                    if (qty / self.depth.lot_size()).round() <= 0.0 {
                                        qty = 0.0;
                                    }
                                    if self.own_fill_depletion {
                                        qty = own_fill_available(
                                            qty,
                                            self.consumed_ask.get(&t).copied().unwrap_or(0.0),
                                            self.own_fill_depletion_persists,
                                            self.depth.lot_size(),
                                        );
                                    }
                                    if qty > 0.0 {
                                        let exec_qty = qty.min(order.leaves_qty);
                                        if self.own_fill_depletion {
                                            *self.consumed_ask.entry(t).or_insert(0.0) += exec_qty;
                                        }
                                        self.fill::<true>(order, timestamp, false, t, exec_qty)?;
                                    }
                                    if order.status == Status::Filled {
                                        return Ok(());
                                    }
                                }
                                order.status = Status::Expired;
                                order.exch_timestamp = timestamp;
                                Ok(())
                            }
                            TimeInForce::GTC => {
                                // Takes the market.
                                for t in self.depth.best_ask_tick()..order.price_tick {
                                    let qty = self.depth.ask_qty_at_tick(t);
                                    if qty > 0.0 {
                                        let exec_qty = qty.min(order.leaves_qty);
                                        self.fill::<true>(order, timestamp, false, t, exec_qty)?;
                                    }
                                    if order.status == Status::Filled {
                                        return Ok(());
                                    }
                                }

                                // The buy order cannot remain in the ask book, as it cannot affect the
                                // market depth during backtesting based on market-data replay. So, even
                                // though it simulates partial fill, if the order size is not small enough,
                                // it introduces unreality.
                                let (price_tick, leaves_qty) = (order.price_tick, order.leaves_qty);
                                self.fill::<true>(order, timestamp, false, price_tick, leaves_qty)
                            }
                            TimeInForce::Unsupported => Err(BacktestError::InvalidOrderRequest),
                        }
                    } else {
                        match order.time_in_force {
                            TimeInForce::GTC | TimeInForce::GTX => {
                                // Initializes the order's queue position.
                                self.queue_model.new_order(order, &self.depth);
                                order.status = Status::New;
                                // The exchange accepts this order.
                                self.buy_orders
                                    .entry(order.price_tick)
                                    .or_default()
                                    .insert(order.order_id);

                                order.exch_timestamp = timestamp;
                                self.orders
                                    .borrow_mut()
                                    .insert(order.order_id, order.clone());
                                Ok(())
                            }
                            TimeInForce::FOK | TimeInForce::IOC => {
                                order.status = Status::Expired;
                                order.exch_timestamp = timestamp;
                                Ok(())
                            }
                            TimeInForce::Unsupported => Err(BacktestError::InvalidOrderRequest),
                        }
                    }
                }
                OrdType::Market => {
                    // todo: set the proper upper bound.
                    for t in self.depth.best_ask_tick()..(self.depth.best_ask_tick() + 100) {
                        let qty = self.depth.ask_qty_at_tick(t);
                        if qty > 0.0 {
                            let exec_qty = qty.min(order.leaves_qty);
                            self.fill::<true>(order, timestamp, false, t, exec_qty)?;
                        }
                        if order.status == Status::Filled {
                            return Ok(());
                        }
                    }
                    order.status = Status::Expired;
                    order.exch_timestamp = timestamp;
                    Ok(())
                }
                OrdType::Unsupported => Err(BacktestError::InvalidOrderRequest),
            }
        } else {
            match order.order_type {
                OrdType::Limit => {
                    // Checks if the sell order price is less than or equal to the current best bid.
                    if order.price_tick <= self.depth.best_bid_tick() {
                        match order.time_in_force {
                            TimeInForce::GTX => {
                                order.status = Status::Expired;
                                order.exch_timestamp = timestamp;
                                Ok(())
                            }
                            TimeInForce::FOK => {
                                // The order must be executed immediately in its entirety; otherwise, the
                                // entire order will be cancelled.
                                let mut execute = false;
                                let mut cum_qty = 0f64;
                                for t in (order.price_tick..=self.depth.best_bid_tick()).rev() {
                                    cum_qty += self.depth.bid_qty_at_tick(t);
                                    if (cum_qty / self.depth.lot_size()).round()
                                        >= (order.qty / self.depth.lot_size()).round()
                                    {
                                        execute = true;
                                        break;
                                    }
                                }
                                if execute {
                                    for t in (order.price_tick..=self.depth.best_bid_tick()).rev() {
                                        let qty = self.depth.bid_qty_at_tick(t);
                                        if qty > 0.0 {
                                            let exec_qty = qty.min(order.leaves_qty);
                                            self.fill::<true>(
                                                order, timestamp, false, t, exec_qty,
                                            )?;
                                            if order.status == Status::Filled {
                                                return Ok(());
                                            }
                                        }
                                    }
                                    unreachable!();
                                } else {
                                    order.status = Status::Expired;
                                    order.exch_timestamp = timestamp;
                                    Ok(())
                                }
                            }
                            TimeInForce::IOC => {
                                // The order must be executed immediately.
                                for t in (order.price_tick..=self.depth.best_bid_tick()).rev() {
                                    // AB#3465 (gentilly): displayed quantity our crossed resting orders already
                                    // took is gone for our IOC orders too.
                                    let mut qty = self.depth.bid_qty_at_tick(t)
                                        - self.crossed_bid.get(&t).copied().unwrap_or(0.0);
                                    if (qty / self.depth.lot_size()).round() <= 0.0 {
                                        qty = 0.0;
                                    }
                                    if self.own_fill_depletion {
                                        qty = own_fill_available(
                                            qty,
                                            self.consumed_bid.get(&t).copied().unwrap_or(0.0),
                                            self.own_fill_depletion_persists,
                                            self.depth.lot_size(),
                                        );
                                    }
                                    if qty > 0.0 {
                                        let exec_qty = qty.min(order.leaves_qty);
                                        if self.own_fill_depletion {
                                            *self.consumed_bid.entry(t).or_insert(0.0) += exec_qty;
                                        }
                                        self.fill::<true>(order, timestamp, false, t, exec_qty)?;
                                    }
                                    if order.status == Status::Filled {
                                        return Ok(());
                                    }
                                }
                                order.status = Status::Expired;
                                order.exch_timestamp = timestamp;
                                Ok(())
                            }
                            TimeInForce::GTC => {
                                // Takes the market.
                                for t in (order.price_tick..=self.depth.best_bid_tick()).rev() {
                                    let qty = self.depth.bid_qty_at_tick(t);
                                    if qty > 0.0 {
                                        let exec_qty = qty.min(order.leaves_qty);
                                        self.fill::<true>(order, timestamp, false, t, exec_qty)?;
                                    }
                                    if order.status == Status::Filled {
                                        return Ok(());
                                    }
                                }

                                // The sell order cannot remain in the bid book, as it cannot affect the
                                // market depth during backtesting based on market-data replay. So, even
                                // though it simulates partial fill, if the order size is not small enough,
                                // it introduces unreality.
                                let (price_tick, leaves_qty) = (order.price_tick, order.leaves_qty);
                                self.fill::<true>(order, timestamp, false, price_tick, leaves_qty)
                            }
                            _ => {
                                unreachable!();
                            }
                        }
                    } else {
                        match order.time_in_force {
                            TimeInForce::GTC | TimeInForce::GTX => {
                                // Initializes the order's queue position.
                                self.queue_model.new_order(order, &self.depth);
                                order.status = Status::New;
                                // The exchange accepts this order.
                                self.sell_orders
                                    .entry(order.price_tick)
                                    .or_default()
                                    .insert(order.order_id);

                                order.exch_timestamp = timestamp;
                                self.orders
                                    .borrow_mut()
                                    .insert(order.order_id, order.clone());
                                Ok(())
                            }
                            TimeInForce::FOK | TimeInForce::IOC => {
                                order.status = Status::Expired;
                                order.exch_timestamp = timestamp;
                                Ok(())
                            }
                            TimeInForce::Unsupported => Err(BacktestError::InvalidOrderRequest),
                        }
                    }
                }
                OrdType::Market => {
                    // todo: set the proper lower bound.
                    for t in ((self.depth.best_bid_tick() - 100)..=self.depth.best_bid_tick()).rev()
                    {
                        let qty = self.depth.bid_qty_at_tick(t);
                        if qty > 0.0 {
                            let exec_qty = qty.min(order.leaves_qty);
                            self.fill::<true>(order, timestamp, false, t, exec_qty)?;
                        }
                        if order.status == Status::Filled {
                            return Ok(());
                        }
                    }
                    order.status = Status::Expired;
                    order.exch_timestamp = timestamp;
                    Ok(())
                }
                OrdType::Unsupported => Err(BacktestError::InvalidOrderRequest),
            }
        }
    }

    fn ack_cancel(&mut self, order: &mut Order, timestamp: i64) -> Result<(), BacktestError> {
        let exch_order = {
            let mut order_borrowed = self.orders.borrow_mut();
            order_borrowed.remove(&order.order_id)
        };

        if exch_order.is_none() {
            order.req = Status::Rejected;
            order.exch_timestamp = timestamp;
            return Ok(());
        }

        let exch_order = exch_order.unwrap();
        let _ = std::mem::replace(order, exch_order);

        // Deletes the order.
        self.arrived.remove(&order.order_id);
        self.unladder(order.side, order.price_tick, order.order_id);
        order.status = Status::Canceled;
        order.exch_timestamp = timestamp;
        Ok(())
    }

    fn ack_modify<const RESET_QUEUE_POS: bool>(
        &mut self,
        order: &mut Order,
        timestamp: i64,
    ) -> Result<(), BacktestError> {
        let (prev_order_price_tick, prev_leaves_qty) = {
            let order_borrowed = self.orders.borrow();
            let exch_order = order_borrowed.get(&order.order_id);

            // The order can be already deleted due to fill or expiration.
            if exch_order.is_none() {
                order.req = Status::Rejected;
                order.exch_timestamp = timestamp;
                return Ok(());
            }

            let exch_order = exch_order.unwrap();
            (exch_order.price_tick, exch_order.leaves_qty)
        };

        // The initialization of the order queue position may not occur when the modified quantity
        // is smaller than the previous quantity, depending on the exchanges. It may need to
        // implement exchange-specific specialization.
        if RESET_QUEUE_POS
            || prev_order_price_tick != order.price_tick
            || order.qty > prev_leaves_qty
        {
            let mut cancel_order = order.clone();
            self.ack_cancel(&mut cancel_order, timestamp)?;
            self.ack_new(order, timestamp)?;
            // todo: Status::Replaced or Status::New?
        } else {
            let mut order_borrowed = self.orders.borrow_mut();
            let exch_order = order_borrowed.get_mut(&order.order_id);
            let exch_order = exch_order.unwrap();

            exch_order.qty = order.qty;
            exch_order.leaves_qty = order.qty;
            exch_order.exch_timestamp = timestamp;
            // todo: Status::Replaced or Status::New?
            exch_order.status = Status::New;
            order.leaves_qty = order.qty;
            order.exch_timestamp = timestamp;
            order.status = Status::New;
        }
        Ok(())
    }
}

impl<AT, LM, QM, MD, FM> PartialFillExchange<AT, LM, QM, MD, FM>
where
    AT: AssetType,
    LM: LatencyModel,
    QM: QueueModel<MD>,
    MD: MarketDepth + L2MarketDepth,
    FM: FeeModel,
{
    fn process_event(&mut self, event: &Event) -> Result<(), BacktestError> {
        if event.is(EXCH_BID_DEPTH_CLEAR_EVENT) {
            self.depth.clear_depth(Side::Buy, event.px);
            self.consumed_bid.clear();
            self.bid_rebuild = Some(None);
        } else if event.is(EXCH_ASK_DEPTH_CLEAR_EVENT) {
            self.depth.clear_depth(Side::Sell, event.px);
            self.consumed_ask.clear();
            self.ask_rebuild = Some(None);
        } else if event.is(EXCH_DEPTH_CLEAR_EVENT) {
            self.depth.clear_depth(Side::None, 0.0);
            self.consumed_bid.clear();
            self.consumed_ask.clear();
            self.bid_rebuild = Some(None);
            self.ask_rebuild = Some(None);
        } else if event.is(EXCH_BID_DEPTH_EVENT) || event.is(EXCH_BID_DEPTH_SNAPSHOT_EVENT) {
            match self.bid_rebuild {
                Some(None) => self.bid_rebuild = Some(Some(event.exch_ts)),
                Some(Some(rebuild_ts))
                    if event.exch_ts > rebuild_ts && !event.is(EXCH_BID_DEPTH_SNAPSHOT_EVENT) =>
                {
                    let depth = &self.depth;
                    self.crossed_bid
                        .retain(|tick, _| depth.bid_qty_at_tick(*tick) > 0.0);
                    self.bid_rebuild = None;
                }
                _ => {}
            }
            let (price_tick, _prev_best_bid_tick, _best_bid_tick, prev_qty, new_qty, timestamp) =
                self.depth
                    .update_bid_depth(event.px, event.qty, event.exch_ts);
            own_fills_after_level_update(
                &mut self.consumed_bid,
                self.own_fill_depletion_persists,
                price_tick,
                self.depth.bid_qty_at_tick(price_tick),
            );
            self.on_bid_qty_chg(price_tick, prev_qty, new_qty, event.ival);
            own_fills_after_level_update(
                &mut self.crossed_bid,
                true,
                price_tick,
                self.depth.bid_qty_at_tick(price_tick),
            );
            self.match_crossing(Side::Sell, timestamp)?;
        } else if event.is(EXCH_ASK_DEPTH_EVENT) || event.is(EXCH_ASK_DEPTH_SNAPSHOT_EVENT) {
            match self.ask_rebuild {
                Some(None) => self.ask_rebuild = Some(Some(event.exch_ts)),
                Some(Some(rebuild_ts))
                    if event.exch_ts > rebuild_ts && !event.is(EXCH_ASK_DEPTH_SNAPSHOT_EVENT) =>
                {
                    let depth = &self.depth;
                    self.crossed_ask
                        .retain(|tick, _| depth.ask_qty_at_tick(*tick) > 0.0);
                    self.ask_rebuild = None;
                }
                _ => {}
            }
            let (price_tick, _prev_best_ask_tick, _best_ask_tick, prev_qty, new_qty, timestamp) =
                self.depth
                    .update_ask_depth(event.px, event.qty, event.exch_ts);
            own_fills_after_level_update(
                &mut self.consumed_ask,
                self.own_fill_depletion_persists,
                price_tick,
                self.depth.ask_qty_at_tick(price_tick),
            );
            self.on_ask_qty_chg(price_tick, prev_qty, new_qty, event.ival);
            own_fills_after_level_update(
                &mut self.crossed_ask,
                true,
                price_tick,
                self.depth.ask_qty_at_tick(price_tick),
            );
            self.match_crossing(Side::Buy, timestamp)?;
        } else if event.is(EXCH_BUY_TRADE_EVENT) {
            let price_tick = (event.px / self.depth.tick_size()).round() as i64;
            self.match_trade(Side::Sell, price_tick, event.qty, event.exch_ts, trade_time(event))?;
        } else if event.is(EXCH_SELL_TRADE_EVENT) {
            let price_tick = (event.px / self.depth.tick_size()).round() as i64;
            self.match_trade(Side::Buy, price_tick, event.qty, event.exch_ts, trade_time(event))?;
        }

        Ok(())
    }
}

impl<AT, LM, QM, MD, FM> Processor for PartialFillExchange<AT, LM, QM, MD, FM>
where
    AT: AssetType,
    LM: LatencyModel,
    QM: QueueModel<MD>,
    MD: MarketDepth + L2MarketDepth,
    FM: FeeModel,
{
    fn event_seen_timestamp(&self, event: &Event) -> Option<i64> {
        event.is(EXCH_EVENT).then_some(event.exch_ts)
    }

    fn process_as_of(&mut self, event: &Event, as_of: i64) -> Result<(), BacktestError> {
        self.as_of = Some(as_of);
        let processed = self.process(event);
        self.as_of = None;
        processed
    }

    fn process(&mut self, event: &Event) -> Result<(), BacktestError> {
        // AB#3465 (gentilly): post-only orders that arrived by this event's time have met their book.
        self.settle_post_only_arrivals(self.as_of.unwrap_or(event.exch_ts), event.exch_ts);
        self.process_event(event)?;
        if let Some(as_of) = self.as_of {
            self.record_post_only_verdicts(as_of);
        }
        Ok(())
    }

    fn process_recv_order(
        &mut self,
        timestamp: i64,
        _wait_resp_order_id: Option<OrderId>,
    ) -> Result<bool, BacktestError> {
        while let Some(mut order) = self.order_e2l.receive(timestamp) {
            // Processes a new order.
            if order.req == Status::New {
                order.req = Status::None;
                self.ack_new(&mut order, timestamp)?;
            }
            // Processes a cancel order.
            else if order.req == Status::Canceled {
                order.req = Status::None;
                self.ack_cancel(&mut order, timestamp)?;
            }
            // Processes a modify order.
            else if order.req == Status::Replaced {
                order.req = Status::None;
                self.ack_modify::<false>(&mut order, timestamp)?;
            } else {
                return Err(BacktestError::InvalidOrderRequest);
            }
            // Makes the response.
            self.order_e2l.respond(order);
        }
        Ok(false)
    }

    fn earliest_recv_order_timestamp(&self) -> i64 {
        self.order_e2l
            .earliest_recv_order_timestamp()
            .unwrap_or(i64::MAX)
    }

    fn earliest_send_order_timestamp(&self) -> i64 {
        self.order_e2l
            .earliest_send_order_timestamp()
            .unwrap_or(i64::MAX)
    }
}

#[cfg(test)]
mod own_fill_depletion_tests {
    use std::collections::HashMap;

    use super::{own_fill_available, own_fills_after_level_update};

    fn taken(qty: f64) -> HashMap<i64, f64> {
        HashMap::from([(834_124, qty)])
    }

    #[test]
    fn a_feed_update_makes_the_level_whole_by_default() {
        let mut consumed = taken(0.03);
        own_fills_after_level_update(&mut consumed, false, 834_124, 0.045);
        assert!(consumed.is_empty());
    }

    #[test]
    fn persisting_keeps_what_we_took_across_a_feed_update() {
        let mut consumed = taken(0.03);
        own_fills_after_level_update(&mut consumed, true, 834_124, 0.045);
        assert_eq!(consumed[&834_124], 0.03);
        // Another level's update leaves it alone.
        own_fills_after_level_update(&mut consumed, true, 834_125, 0.0);
        assert_eq!(consumed[&834_124], 0.03);
    }

    #[test]
    fn persisting_caps_what_we_took_at_the_displayed_quantity() {
        let mut consumed = taken(0.03);
        own_fills_after_level_update(&mut consumed, true, 834_124, 0.01);
        assert_eq!(consumed[&834_124], 0.01);
        // Quantity added afterwards is available beyond what we took.
        own_fills_after_level_update(&mut consumed, true, 834_124, 0.05);
        assert_eq!(consumed[&834_124], 0.01);
    }

    #[test]
    fn a_fully_taken_level_offers_nothing_whatever_rounding_the_fills_left() {
        // 3.06427 displayed; fills of 0.94783 and of the rest add up to one unit in the last place short of it.
        let (displayed, first) = (3.06427_f64, 0.94783_f64);
        let taken = first + (displayed - first);
        assert!(
            displayed - taken > 0.0,
            "the case must leave a positive remainder"
        );
        assert_eq!(own_fill_available(displayed, taken, true, 0.00001), 0.0);
        // A real remainder is available in full, and the default rule is the plain difference.
        assert_eq!(own_fill_available(1.0, 0.6, true, 0.00001), 1.0 - 0.6);
        assert_eq!(
            own_fill_available(displayed, taken, false, 0.00001),
            displayed - taken
        );
    }

    #[test]
    fn persisting_starts_fresh_once_the_feed_empties_the_level() {
        let mut consumed = taken(0.03);
        own_fills_after_level_update(&mut consumed, true, 834_124, 0.0);
        assert!(consumed.is_empty());
    }
}
