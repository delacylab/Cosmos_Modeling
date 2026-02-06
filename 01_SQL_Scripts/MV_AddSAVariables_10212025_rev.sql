DROP TABLE IF EXISTS #EncounterList
SELECT *
INTO #EncounterList
FROM PROJECTS.[ProjectD52021F].[dbo].[BipolarDepression_EncLevelVars_v2]
;

-- Any encounter 30 days prior to the encounter in question
DROP TABLE IF EXISTS #AnyEncounter30DaysPrior
SELECT DISTINCT el.EncounterKey
INTO #AnyEncounter30DaysPrior
FROM #EncounterList el
JOIN dbo.EncounterFact e
	ON el.PatientDurableKey = e.PatientDurableKey
	AND e.Date < el.EncDate
	AND e.Date >= DATEADD(DAY, -30, el.EncDate)
;

-- All suicide attempts for the patients in my cohort
DROP TABLE IF EXISTS #PatientList
SELECT PatientDurableKey
INTO #PatientList
FROM PROJECTS.[ProjectD52021F].[dbo].[BipolarDepression_PatientLevelVars]
;

DROP TABLE IF EXISTS #SuicideAttemptRaw
SELECT DISTINCT
	dx.EncounterKey
INTO #SuicideAttemptRaw
FROM dbo.DiagnosisEventFact dx
JOIN #PatientList d	-- I only want suicide attempt data for the patients in the depression+ list, who are already vetted valid, to hopefully speed the query up
	ON dx.PatientDurableKey = d.PatientDurableKey
JOIN PROJECTS.ProjectD0F7BC.dbo.MV_SuicideAttempt_ICD10_CodeList_v2 cl
	ON dx.DiagnosisKey = cl.DiagnosisKey
	AND dx.Type IN  (N'Billing Final Diagnosis', N'Encounter Diagnosis', N'Billing Admission Diagnosis', N'Discharge Diagnosis', N'Admitting Diagnosis') 
	AND dx.StartDateKey >= 20140101
	AND dx.Count = 1
GO
DECLARE @timestamp nvarchar(50)
SET @timestamp = convert(nvarchar(50), GETDATE(), 121)
RAISERROR('%s: Finished pulling raw data suicide attempt', 10, 1, @timestamp) WITH NOWAIT
GO

DROP TABLE IF EXISTS #SuicideAttemptEnc
SELECT DISTINCT
	e.PatientDurableKey,
	e.EncounterKey,
	CAST(e.Date AS date) AS EncDate	-- I  only want the date part from the ecouunter date (no time)
INTO #SuicideAttemptEnc
FROM #SuicideAttemptRaw r
JOIN dbo.EncounterFact e	-- I am doing this JOIN to make sure the encounter is valid
	ON r.EncounterKey = e.EncounterKey
	AND e.Count = 1	
GO
;

DROP TABLE IF EXISTS #TwoNewVariables
SELECT DISTINCT
	v2.PatientDurableKey,
	v2.EncounterKey,
	v2.EncDate,
	CASE 
		WHEN a30.EncounterKey IS NULL THEN NULL
		WHEN sa30.EncounterKey IS NOT NULL THEN 'Y'
		ELSE 'N'
	END AS HxSuicideAttempt30DaysPrior
INTO #TwoNewVariables
FROM #EncounterList v2
LEFT JOIN #AnyEncounter30DaysPrior a30
	ON v2.EncounterKey = a30.EncounterKey
LEFT JOIN #SuicideAttemptEnc sa30
	ON sa30.EncDate < v2.EncDate
	AND sa30.EncDate >= DATEADD(DAY, -30, v2.EncDate)
	AND v2.PatientDurableKey = sa30.PatientDurableKey
;

INSERT INTO PROJECTS.ProjectD52021F.[dbo].[BipolarDepression_AddlVarsOct2025]
SELECT * FROM #TwoNewVariables
;