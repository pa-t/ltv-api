SELECT order_id::integer, customer_id::integer , afid::text, campaign_id::integer, cc_type::text, main_product_id::integer,
       DATE(time_stamp) time_stamp, recurring_date, billing_cycle, order_status::integer,
       order_total, is_refund::boolean, on_hold::boolean, hold_date, billing_state::text, is_fraud::boolean, is_chargeback::boolean
FROM public.sticky_brainable_order_stg
WHERE order_status::integer NOT IN (7, 11)
  AND campaign_id::integer IN (2, 3, 7, 9, 30, 33, 63, 81, 82, 83, 84, 89, 90, 91,
                               92, 121, 122, 171, 172, 173, 174, 175, 176)
ORDER BY customer_id ASC, main_product_id ASC, billing_cycle ASC;