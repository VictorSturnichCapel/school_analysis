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

SELECT corr(coluna_x, coluna_y) AS correlacao
FROM sua_tabela;

select student_id, subject, min_scr as first_score, max_scr as latest_score  from (
select student_id, subject, score, exam_date,
FIRST_VALUE(score) OVER(PARTITION BY student_id, subject ORDER BY exam_date asc) as min_scr,
LAST_VALUE(score) OVER(PARTITION BY student_id, subject ORDER BY exam_date asc) as max_scr,
ROW_NUMBER() OVER(PARTITION BY student_id, subject ORDER BY exam_date desc) as rn
from Scores) as q
where rn = 1 and min_scr < max_scr

SELECT 
    (SUM((x - avg_x) * (y - avg_y)) /
    SQRT(SUM(POW(x - avg_x, 2)) * SUM(POW(y - avg_y, 2)))) AS correlation
FROM (
    SELECT 
        x_column AS x,
        y_column AS y,
        (SELECT AVG(x_column) FROM your_table) AS avg_x,
        (SELECT AVG(y_column) FROM your_table) AS avg_y
    FROM your_table
) AS subquery;

SELECT 
    (SUM((x - avg_x) * (y - avg_y)) /
    SQRT(SUM(POW(x - avg_x, 2)) * SUM(POW(y - avg_y, 2)))) AS correlation
FROM (
    SELECT 
        student_id  AS x,
        score  AS y,
        (SELECT AVG(student_id) FROM Scores) AS avg_x,
        (SELECT AVG(score ) FROM Scores) AS avg_y
    FROM Scores 
) AS subquery;