-- 1) Create staging table
DROP TABLE IF EXISTS staging_workorders;

CREATE TABLE staging_workorders (
    wo_id TEXT,
    wo_description TEXT,
    collection TEXT,
    primary_code TEXT,
    primary_code_description TEXT,
    secondary_code TEXT,
    secondary_code_description TEXT,
    equipment_id TEXT,
    equipment_name TEXT,
    preventive TEXT,
    work_priority TEXT,
    actual_finish TEXT,
    duration TEXT,
    actual_labor_hours TEXT
);

-- 2) Load CSV into staging
COPY staging_workorders
FROM '/tmp/workorders.csv'
WITH (FORMAT csv, HEADER true, DELIMITER E',');


-- 3) Insert/Upsert into assets
INSERT INTO assets (assetnum, description)
SELECT DISTINCT equipment_id, equipment_name
FROM staging_workorders
WHERE equipment_id IS NOT NULL AND equipment_id <> ''
ON CONFLICT (assetnum) DO NOTHING;

-- 4) Insert/Upsert into jobplans (primary + secondary)
INSERT INTO jobplans (plannum, description)
SELECT DISTINCT code, description
FROM (
    SELECT primary_code AS code, primary_code_description AS description
    FROM staging_workorders
    UNION
    SELECT secondary_code, secondary_code_description
    FROM staging_workorders
) t
WHERE code IS NOT NULL AND code <> ''
ON CONFLICT (plannum) DO NOTHING;

-- 5) Insert/Upsert into workorders
INSERT INTO workorders (workordernum, assetid, type, priority, description, enddate,startdate, actual_finish)

SELECT 
    s.wo_id,
    a.assetid,
    CASE 
        WHEN s.preventive::BOOLEAN = TRUE THEN 'Preventive'
        ELSE 'Corrective'
    END AS type,
    NULLIF(s.work_priority,'')::INT,
    s.wo_description,

    to_timestamp(s.actual_finish, 'MM/DD/YY HH24:MI') AS enddate,
    to_timestamp(s.actual_finish, 'MM/DD/YY HH24:MI') 
        - s.duration::interval AS startdate,

    NULLIF(s.actual_finish,'')::DATE

FROM staging_workorders s
LEFT JOIN assets a ON a.assetnum = s.equipment_id
ON CONFLICT (workordernum) DO NOTHING;

-- 6) Insert workorder labor with HH:MM:SS -> decimal hours conversion
INSERT INTO workorderlabor (workorderid, hoursworked)
SELECT 
    w.workorderid,
    CASE 
        WHEN s.actual_labor_hours ~ '^\d+:\d{2}:\d{2}$' THEN
            split_part(s.actual_labor_hours, ':', 1)::NUMERIC
            + split_part(s.actual_labor_hours, ':', 2)::NUMERIC / 60
            + split_part(s.actual_labor_hours, ':', 3)::NUMERIC / 3600
        WHEN s.actual_labor_hours ~ '^\d+:\d{2}$' THEN
            split_part(s.actual_labor_hours, ':', 1)::NUMERIC
            + split_part(s.actual_labor_hours, ':', 2)::NUMERIC / 60
        ELSE NULLIF(s.actual_labor_hours,'')::NUMERIC
    END AS hoursworked
FROM staging_workorders s
JOIN workorders w ON w.workordernum = s.wo_id
WHERE s.actual_labor_hours IS NOT NULL AND s.actual_labor_hours <> '';

-- 7) (Optional) Drop staging after successful load
DROP TABLE staging_workorders;
