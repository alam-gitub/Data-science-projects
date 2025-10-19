-- 
Create Database Books_Store;	--Creating Database
drop table  if exists Books;	--Ensuring table availability
Create Table Books(				-- Creating Table
	Book_ID	int	Primary key,
Title	varchar	,
Author	varchar	,
Genre	varchar	,
Published_Year	int	,
Price	float	,
Stock	int	

);
Select * from Books; 		--Checking Created table 

copy Books(					-- Retreiving Data 
Book_ID,
Title,	
Author,
Genre,		
Published_Year ,
Price,		
Stock)	
	From 'C:\Program Files\PostgreSQL\sql project\Books.csv'
	Delimiter ','
	CSV Header;

Select * from Books;		--Checking imported data in table 


Drop table  if exists Customers;	--Ensuring table availability
Create Table Customers(				---- Creating Table
Customer_ID	int Primary key,
Name	varchar,	
Email	varchar	unique,
Phone	int 	unique,
City	varchar,	
Country	varchar	

);

Select * from Customers;		--Ensuring table availability	


Copy Customers(					---- Retreiving Data 
Customer_ID,
Name,	
Email,
Phone,
City,	
Country
)
	from 'C:\Program Files\PostgreSQL\sql project\Customers.csv'
	Delimiter ','
	CSV Header;

Select * from Customers; 					----Checking imported data in table 


Drop table  if exists Orders;				--Ensuring table availability

Create Table Orders(				------ Creating Table
Order_ID	int	Primary key,
Customer_ID	int References Customers(Customer_ID),	
Book_ID	int	References Books(Book_ID),
Order_Date	date,	
Quantity	int	,
Total_Amount	int	
);

Copy Orders (						---- Retreiving Data 
Order_ID,
Customer_ID	,
Book_ID	,
Order_Date,	
Quantity,
Total_Amount
)
	from 'C:\Program Files\PostgreSQL\sql project\Orders.csv'
	Delimiter ','
	CSV Header;

Alter Table Orders 
	Alter Column Total_Amount type float;			--Change Data type because found data type incorrect
	

Select * from Orders;			--Checking imported data in table 
								


-----------------------------TASKS--------------------------------------------

--1. Retrieve all books in the "Fiction" Genre:

Select * From Books;		----Ensuring Column Name 

	Select * 					----Finding Books in Fiction Genre
	From Books
	Where genre = 'Fiction' ;


--2. Find Books published after the year 1950:

Select *						----Finding Books after published 1950
From Books
Where published_year >1950;


--3. List of all customers from the Canada

Select * From Customers; 		----Ensuring Column Name 

Select * 						----Finding customers from Canada
from Customers
Where country = 'Canada';


--4. Show orders placed in November 2023

Select * from Orders;			----Ensuring Column Name 

Select * 						---- Placed order in Month November
From Orders
Where order_date Between '2023-11-01' And '2023-11-30'
Order By order_date;

------------Or----------
Select * 
From Orders
Where order_date >= Date '2023-11-01'
And order_date < Date '2023-12-01';


--5. Retrieve the total stock of books available:

	Select * From Books;		----Ensuring Column Name 

Select 							----Available Stock
sum(stock) As Available_Stock
From Books;


--6. Find the details of the most expensive book:

Select *					----Most Expensive Book
From Books
Order By price desc
limit 1;


--7. Show all customers who ordered more than 1 quantity of a book:

Select * From orders;		----Ensuring Column Name 

SELECT						----Our customers who purchased more than 1 books
	CUSTOMER_ID,
	QUANTITY
FROM
	ORDERS
WHERE
	QUANTITY > 1
ORDER BY
	QUANTITY Desc;


--8. Retrive all orders where the total amount exceed $20:
Select * From orders;		----Ensuring Column Name 

Select * 					----Orders detail which Expensive more than 20
From Orders
Where total_amount > 20
Order by total_amount desc;


--9. List all genre available in the books table:
Select * From Books;		----Ensuring Column Name

Select distinct(genre)		----Retrive all Genre 
from Books;


--10. Find the books with lowest stock:

Select *					----Retrieve minimum stock
from Books
order by stock Asc limit 1;


--11. Calculate the total revenue generated from all orders;
Select * From orders;		----Ensuring Column Name 

select 						---- Retrieve total Revenue
sum(total_amount) AS Revenue	
from Orders;


--12. Retrieve the total number of books sold by each genre:
Select * From orders;		----Ensuring Column Name 
Select * From Books;		----Ensuring Column Name
Select * From Customers; 	----Ensuring Column Name 

Select b.genre,						----Total sold book in every genre
Sum(o.quantity) As Total_Sold
from Books b
join Orders o						----Joining books table with orders table 
	on b.book_id = o.book_id
group by b.genre
Order By Total_Sold desc;


--13. Find the average Revenue of books in the "Fantasy" genre:

Select b.genre,	 					----Average revenue of book in fantasy genre
Avg(o.total_amount) As Total
from Books b
join Orders o						----Joining books table with orders table 
	on b.book_id = o.book_id
Where b.genre = 'Fantasy'
Group by b.genre;


--14. Find the average Price of books in the "Fantasy" genre:

Select Avg(price)As Average_price_Fantasy
from Books
Where genre= 'Fantasy';


--15.List of customers who have placed at least 2 orders:
Select * From orders;		----Ensuring Column Name 
Select * From Books;		----Ensuring Column Name
Select * From Customers; 	----Ensuring Column Name 

Select c.customer_id, 
	c.name, Count(o.order_id)
from Customers c
	join Orders o
	on c.customer_id = o.customer_id
	Group by c.customer_id
Having Count(o.order_id) >=2;


--16. Find the most frequently ordered book

Select o.book_id,
	b.title,
	o.quantity,
	Count(o.order_id) As Total_order
	from Orders o
	Join Books b
    	on b.book_id = o.book_id 
	Group by o.book_id, o.quantity, b.title
	Order by Total_order desc 
limit 1;


--17. Show the top 3 most expensive books of "Fantasy" Genre:
Select * From Books;		----Ensuring Column Name

Select book_id, 
	title, 
	genre, 
	price
From Books
Where genre = 'Fantasy'
Order by price desc
Limit 3;


--18. Retrive the total quantity of books sold by each author:
Select * From Books;		----Ensuring Column Name
Select * From Orders;		----Ensuring Column Name 
Select * From Customers;	----Ensuring Column Name 

Select b.author, 
sum(o.quantity)As Total_Sold
From Orders o
Join Books b
	ON o.book_id = b.book_id
Group By b.author
Order By Total_Sold Desc;


--19. List the cities where customers who spent over $30 are located:

Select Distinct c.city,
sum(o.total_amount)As Total_spend
From Orders o
Join Customers c
	On o.customer_id = c.customer_id
Group By c.city, c.customer_id
Having sum(o.total_amount) > 30;


--20. Find the customer who spent the most on orders:
Select * From Books;		----Ensuring Column Name
Select * From Orders;		----Ensuring Column Name 
Select * From Customers;	----Ensuring Column Name 

Select c.customer_id,
c.name,
sum(o.total_amount) As Total_Expenses
From Customers c 
Join Orders o
	On c.customer_id = o.customer_id
Group By c.customer_id
Order By Total_Expenses Desc
Limit 1;


--21. Calculate the stock remaining after fulfilling all orders:
Select * From Books;		----Ensuring Column Name
Select * From Orders;		----Ensuring Column Name 
Select * From Customers;	----Ensuring Column Name 

Select b.title,
b.book_id
b.stock As Total_Stock, 
Coalesce(sum(o.quantity),0) As Total_Sold,
b.stock - Coalesce(sum(o.quantity),0) As Remaining_stock
From Orders o
Right Join Books b
	On b.book_id = o.book_id
Group By b.book_id
Order By Remaining_stock Asc;




