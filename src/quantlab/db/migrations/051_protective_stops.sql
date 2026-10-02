-- Broker-held protective stops (execution range 050-059).
-- A protective stop is an orders row with purpose='exit', order_type='stop', time_in_force='gtc'
-- (its order_intents row has purpose='protective_stop'). stop_price is the RAW trigger price sent to
-- the broker; NULL for every other order type.
ALTER TABLE orders ADD COLUMN stop_price REAL;
