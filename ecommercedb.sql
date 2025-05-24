--2.1 Basic Queries 
--● Calculate the total sales revenue from all orders. 
SELECT SUM(od.quantity * od.unit_price) AS total_revenue
FROM order_details od;

--● List the top 5 best-selling products by quantity sold. 
SELECT TOP 5 p.name AS product_name, SUM(od.quantity) AS total_quantity_sold
FROM order_details od
JOIN products p ON od.product_id = p.id
GROUP BY p.name
ORDER BY total_quantity_sold DESC;

--● Identify customers with the highest number of orders. 
select top 10 customer_id , count(id) as [Total_orders]
from orders
group by customer_id
order by [Total_orders] desc
--● Generate an alert for products with stock quantities below 20 units. 
select p.id , p.name  , stock_quantity   from products p
where stock_quantity < 20
order by stock_quantity
--● Determine the percentage of orders that used a discount. 
WITH OrderDiscountFlags AS (
SELECT
o.id AS order_id,
-- Flag orders where ANY product had an active discount during the order date
CASE WHEN EXISTS (
SELECT 1
FROM order_details od
INNER JOIN products p ON od.product_id = p.id
INNER JOIN discounts d ON
(d.product_id = p.id OR d.category_id = p.category_id)
WHERE
od.order_id = o.id
AND d.is_active = 1  -- Discount is active
AND o.order_date BETWEEN d.start_date AND d.end_date  -- Valid during order
) THEN 1 ELSE 0 END AS has_discount
FROM orders o
)
SELECT
YEAR(o.order_date) AS OrderYear,
MONTH(o.order_date) AS OrderMonth,
COUNT(o.id) AS TotalOrders,
SUM(odf.has_discount) AS DiscountedOrders,
ROUND(
(SUM(odf.has_discount) * 100.0 / COUNT(o.id)),
2
) AS DiscountPercentage
FROM orders o
LEFT JOIN OrderDiscountFlags odf ON o.id = odf.order_id
GROUP BY
YEAR(o.order_date),
MONTH(o.order_date)
ORDER BY
OrderYear,
OrderMonth;
--● Calculate the average rating for each product.
SELECT
p.id AS ProductID,
p.name AS ProductName,
ROUND(AVG(r.rating), 2) AS AverageRating,
COUNT(r.rating) AS TotalReviews
FROM
products p
LEFT JOIN
reviews r ON p.id = r.product_id
GROUP BY
p.id, p.name
ORDER BY
AverageRating DESC; -- Sort from highest to lowest rated


--.2 Advanced Queries 
--● Compute the 30-day customer retention rate after their first purchase. 
WITH FirstOrders AS (
-- Get each customer's first order date (cohort)
SELECT
customer_id,
MIN(order_date) AS first_order_date
FROM orders
GROUP BY customer_id
),
RetentionData AS (
-- Check if they returned within 30 days
SELECT
fo.customer_id,
fo.first_order_date,
CASE
WHEN EXISTS (
SELECT 1
FROM orders o
WHERE o.customer_id = fo.customer_id
AND o.order_date > fo.first_order_date
AND o.order_date <= DATEADD(DAY, 30, fo.first_order_date)
) THEN 1
ELSE 0
END AS retained_30d
FROM FirstOrders fo
)
SELECT
FORMAT(first_order_date, 'yyyy-MM') AS cohort_month,
COUNT(customer_id) AS total_customers,
SUM(retained_30d) AS retained_customers,
ROUND(SUM(retained_30d) * 100.0 / COUNT(customer_id), 2) AS retention_rate
FROM RetentionData
GROUP BY FORMAT(first_order_date, 'yyyy-MM')
ORDER BY cohort_month;


--● Recommend products frequently bought together with items in customer wishlists. 

WITH WishlistItems AS (
    SELECT DISTINCT w.customer_id, w.product_id
    FROM wishlists w
),
OrderProducts AS (
    SELECT o.customer_id, od.product_id
    FROM orders o
    JOIN order_details od ON o.id = od.order_id
),
FrequentlyBoughtTogether AS (
    SELECT wi.product_id AS wishlist_product, op.product_id AS bought_product, COUNT(*) AS purchase_count
    FROM WishlistItems wi
    JOIN OrderProducts op ON wi.customer_id = op.customer_id
    WHERE op.product_id != wi.product_id -- Exclude the wishlist item itself
    GROUP BY wi.product_id, op.product_id
    HAVING COUNT(*) > 1 -- Filter for items bought together more than once
)
SELECT
    fbt.wishlist_product,
    p1.name AS wishlist_product_name,
    fbt.bought_product,
    p2.name AS bought_product_name,
    fbt.purchase_count
FROM FrequentlyBoughtTogether fbt
JOIN products p1 ON fbt.wishlist_product = p1.id
JOIN products p2 ON fbt.bought_product = p2.id
ORDER BY fbt.purchase_count DESC;

--● Find pairs of products commonly bought together in the same order. 
SELECT
od1.product_id AS Product1,
od2.product_id AS Product2,
COUNT(*) AS TimesBoughtTogether
FROM
order_details od1
JOIN
order_details od2
ON od1.order_id = od2.order_id
AND od1.product_id < od2.product_id  -- avoid duplicates and self-pairs
GROUP BY
od1.product_id, od2.product_id
ORDER BY
TimesBoughtTogether DESC;

--● Calculate the time taken to deliver orders in days. 
select o.id as order_id , DATEDIFF(DAY , o.order_date , s.shipping_date) as DeliveryTimeinDays
from orders o
inner join shipping s
on o.id = s.order_id
where s.status = 'delivered'