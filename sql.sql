select s.score as 'score', dense_rank() over (order by s.score DESC) as 'rank' from Scores as s group by s.id having s.score > 0 order by 'rank'

select q.Department, q.Employee, q.Salary 
from(select d.name as 'Department', e.name as 'Employee', e.salary as 'Salary', dense_rank() OVER(partition by d.id order by e.salary DESC) as 'rank' 
from Employee as e LEFT JOIN Department as d ON e.departmentId  = d.id) as q  where q.rank < 4


select s.name from SalesPerson as s  where s.sales_id  not in (SELECT sales_id from Orders LEFT JOIN Company on Orders.com_id  = Company.com_id where Company.name not like '%RED%' )

select w1.id as 'ID' from Weather as w left join Weather as w1 on w.recordDate = subdate( w1.recordDate, 1)  where w.temperature < w1.temperature

select t.request_at as 'Day', ROUND( SUM(CASE when t.status like 'cancelled%' then 1 else 0 END )/ COUNT(t.id) ,2) as 'Cancellation Rate' from Trips as t LEFT join Users as u 
on t.client_id = u.users_id  LEFT join Users as u1 on t.driver_id = u1.users_id 
where u.banned = 'No' and u1.banned = 'No' AND u.banned IS NOT NULL AND u1.banned IS NOT NULL and request_at between "2013-10-01" and "2013-10-03" group by t.request_at having COUNT(t.id) >= 1 

select
    student_id,
    score,
    percent_rank() over (order by score DESC) *100 as 'percent_rank'
from 
    students
where
    created_at >= CURRENT_DATE - INTERVAL '30 days'
                                           30 DAY

select w1.id as 'ID' from Weather as w left join Weather as w1 on w.recordDate = subdate( w1.recordDate, 1)  where w.temperature < w1.temperature