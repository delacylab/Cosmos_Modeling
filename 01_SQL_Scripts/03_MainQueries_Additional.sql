/*SELECT *
FROM PROJECTS.ProjectD0F7BC.dbo.MV_Conditions_CodeList c
WHERE c.Category = 'Bipolar, active, depressive episode'
;
*/

DECLARE @timestamp nvarchar(50)
SET @timestamp = convert(nvarchar(50), GETDATE(), 121)
RAISERROR('%s: Session started', 10, 1, @timestamp) WITH NOWAIT
GO

drop table if exists #TempBipolarDepression;
SELECT
	dx.EncounterKey
INTO #TempBipolarDepression
FROM dbo.DiagnosisEventFact dx
JOIN PROJECTS.ProjectD0F7BC.dbo.MV_Conditions_CodeList cl
	ON dx.DiagnosisKey = cl.DiagnosisKey
	AND cl.Category = 'Bipolar, active, depressive episode'
	AND cl.ActiveYN = 1
	AND dx.Type IN  (N'Billing Final Diagnosis', N'Encounter Diagnosis', N'Billing Admission Diagnosis', N'Discharge Diagnosis', N'Admitting Diagnosis') 
	AND dx.StartDateKey BETWEEN 20160101 AND 20241231
	AND dx.Count = 1
GO
DECLARE @timestamp nvarchar(50)
SET @timestamp = convert(nvarchar(50), GETDATE(), 121)
RAISERROR('%s: Finished pulling raw data', 10, 1, @timestamp) WITH NOWAIT
GO

DROP TABLE IF EXISTS #TempBipolarDepressionEnc
SELECT DISTINCT
	e.PatientDurableKey,
	e.EncounterKey,
	e.DateKey AS EncDateKey,
	e.Date AS EncDate,
	e.AgeKey AS EncAgeKey,
	e.MarkedSelfPay,
	e.PrimaryCoverageFinancialClass_X
INTO #TempBipolarDepressionEnc
FROM #TempBipolarDepression t
JOIN COSMOS.dbo.EncounterFact e
	ON t.EncounterKey = e.EncounterKey
	AND e.Count = 1
JOIN COSMOS.dbo.PatientDim p
	ON e.PatientDurableKey = p.DurableKey
	AND p.IsCurrent=1
	AND p.IsValid=1
	AND p.UseInCosmosAnalytics_X=1
GOFrom 
DECLARE @timestamp nvarchar(50)
SET @timestamp = convert(nvarchar(50), GETDATE(), 121)
RAISERROR('%s: Finished creating encounter list', 10, 1, @timestamp) WITH NOWAIT
GO

-- 4,243,353 encounters
SELECT COUNT_BIG(DISTINCT EncounterKey) FROM #TempBipolarDepressionEnc
GO

DROP TABLE IF EXISTS #TempBipolarDepressionPats
SELECT DISTINCT
	e.PatientDurableKey,
	p.BirthDate,
	p.DeathDate,
	p.FirstRace,
	p.SecondRace,
	p.ThirdRace,
	p.FourthRace,
	p.FifthRace,
	p.Ethnicity,
	p.Sex,
	p.SexAssignedAtBirth,
	p.GenderIdentity,
	p.PreferredLanguage,
	p.ValidatedStateOrProvince_X,
	p.ValidatedStateOrProvinceAbbreviation_X,
	p.MaritalStatus,
	p.SviHouseholdCharacteristicsPctlRankByZip2020_X,
	p.SviHouseholdCompositionPctlRankingByZip2018_X,
	p.SviHousingTypeTransportationPctlRankByZip2020_X,
	p.SviHousingTypeTransportationPctlRankingByZip2018_X,
	p.SviMinorityStatusLanguagePctlRankingByZip2018_X,
	p.SviOverallPctlRankByZip2020_X,
	p.SviOverallPctlRankingByZip2018_X,
	p.SviRacialEthnicMinorityStatusPctlRankByZip2020_X,
	p.SviSocioeconomicPctlRankByZip2020_X,
	p.SviSocioeconomicPctlRankingByZip2018_X
INTO #TempBipolarDepressionPats
FROM #TempBipolarDepressionEnc e
JOIN COSMOS.dbo.PatientDim p
	ON e.PatientDurableKey = p.DurableKey
	AND p.IsCurrent = 1
	AND p.IsValid = 1
	AND p.UseInCosmosAnalytics_X = 1
GO
DECLARE @timestamp nvarchar(50)
SET @timestamp = convert(nvarchar(50), GETDATE(), 121)
RAISERROR('%s: Finished creating patient list', 10, 1, @timestamp) WITH NOWAIT
GO

-- 578,521 unique patients
SELECT COUNT_BIG(DISTINCT PatientDurableKey) FROM #TempBipolarDepressionPats
GO

-- Flowsheet data
DROP TABLE IF EXISTS #TempBipolarDepressionFlowsheetData
SELECT
	PatientDurableKey,
	FlowsheetRowKey,
	DateKey,
	Value,
	NumericValue
INTO #TempBipolarDepressionFlowsheetData
FROM (
SELECT DISTINCT
	coh.PatientDurableKey,
	fv.FlowsheetRowKey,
	fv.DateKey,
	fv.Value,
	fv.NumericValue,
	ROW_NUMBER() OVER (PARTITION BY coh.PatientDurableKey, fv.FlowsheetRowKey ORDER BY fv.DateKey DESC) AS Line
FROM #TempBipolarDepressionPats coh
JOIN COSMOS.dbo.FlowsheetValueFact fv
	ON coh.PatientDurableKey = fv.PatientDurableKey
	AND fv.Count = 1
	AND fv.Value IS NOT NULL
	AND fv.Value NOT IN ('*Masked', '*Unspecified', '*Deleted', '*Not Applicable')
) q
WHERE Line = 1	-- I want the most recent value for each patient
GO
DECLARE @timestamp nvarchar(50)
SET @timestamp = convert(nvarchar(50), GETDATE(), 121)
RAISERROR('%s: Finished flowsheet data', 10, 1, @timestamp) WITH NOWAIT
GO

-- Hospice
DROP TABLE IF EXISTS #TempBipolarDepressionHospice
SELECT *
INTO #TempBipolarDepressionHospice
FROM (
SELECT 
	dx.PatientDurableKey
FROM COSMOS.dbo.DiagnosisEventFact dx
JOIN COSMOS.dbo.DiagnosisTerminologyDim dt
	ON dx.DiagnosisKey = dt.DiagnosisKey
	AND dt.Type = 'ICD-10-CM'
	AND dt.Value = 'Z51.5'
JOIN #TempBipolarDepressionPats p
	ON p.PatientDurableKey = dx.PatientDurableKey
	AND dx.Type IN  (N'Billing Final Diagnosis', N'Encounter Diagnosis', N'Billing Admission Diagnosis', N'Discharge Diagnosis', N'Admitting Diagnosis') 
	AND dx.Count = 1

UNION

SELECT
	p.PatientDurableKey
FROM [COSMOS].[dbo].[ProcedureEventFact] pe
JOIN COSMOS.dbo.ProcedureDim pr
	ON pe.ProcedureDurableKey = pr.DurableKey
	AND pr.ShortName LIKE '%hospice%'
	AND pr.Category NOT LIKE 'LOINC%'
	AND pe.Count = 1
JOIN #TempBipolarDepressionPats p
	ON p.PatientDurableKey = pe.PatientDurableKey
) q
GO
DECLARE @timestamp nvarchar(50)
SET @timestamp = convert(nvarchar(50), GETDATE(), 121)
RAISERROR('%s: Finished hospice', 10, 1, @timestamp) WITH NOWAIT
GO


-- Comorbidities
DROP TABLE IF EXISTS #TempBipolarDepressionComorbidities
SELECT DISTINCT
	dx.PatientDurableKey,
	cl.Category
INTO #TempBipolarDepressionComorbidities
FROM COSMOS.dbo.DiagnosisEventFact dx
JOIN PROJECTS.ProjectD0F7BC.dbo.MV_SuicideAttempt_ICD10_Comorbidities_CodeList cl
	ON dx.DiagnosisKey = cl.DiagnosisKey
	AND dx.Type IN  (N'Billing Final Diagnosis', N'Encounter Diagnosis', N'Problem List', N'Billing Admission Diagnosis', N'Discharge Diagnosis', N'Admitting Diagnosis') 
	AND dx.Count = 1
JOIN #TempBipolarDepressionPats p
	ON p.PatientDurableKey = dx.PatientDurableKey
GO
DECLARE @timestamp nvarchar(50)
SET @timestamp = convert(nvarchar(50), GETDATE(), 121)
RAISERROR('%s: Finished comorbidities', 10, 1, @timestamp) WITH NOWAIT
GO

-- Suicide attempt
DROP TABLE IF EXISTS #TempSuicideAttemptRaw
SELECT DISTINCT
	dx.EncounterKey,
	cl.Category
INTO #TempSuicideAttemptRaw
FROM COSMOS.dbo.DiagnosisEventFact dx
JOIN #TempBipolarDepressionPats d	-- I only want suicide attempt data for the patients in the depression+ list, who are already vetted valid, to hopefully speed the query up
	ON dx.PatientDurableKey = d.PatientDurableKey
JOIN PROJECTS.ProjectD0F7BC.dbo.MV_SuicideAttempt_ICD10_CodeList_v2 cl
	ON dx.DiagnosisKey = cl.DiagnosisKey
	AND dx.Type IN  (N'Billing Final Diagnosis', N'Encounter Diagnosis', N'Billing Admission Diagnosis', N'Discharge Diagnosis', N'Admitting Diagnosis') 
	AND dx.StartDateKey >= 20150101
	AND dx.Count = 1
GO
DECLARE @timestamp nvarchar(50)
SET @timestamp = convert(nvarchar(50), GETDATE(), 121)
RAISERROR('%s: Finished pulling raw data suicide attempt', 10, 1, @timestamp) WITH NOWAIT
GO

DROP TABLE IF EXISTS #TempSuicideAttemptEnc
SELECT DISTINCT
	e.PatientDurableKey,
	e.EncounterKey,
	e.DateKey AS EncDateKey,
	CAST(e.Date AS date) AS EncDate,	-- I  only want the date part from the ecouunter date (no time)
	e.AgeKey,
	r.Category
INTO #TempSuicideAttemptEnc
FROM #TempSuicideAttemptRaw r
JOIN COSMOS.dbo.EncounterFact e
	ON r.EncounterKey = e.EncounterKey
	AND e.Count = 1
GO

DROP TABLE IF EXISTS #TempSuicideAttemptPats
SELECT DISTINCT
	e.PatientDurableKey
INTO #TempSuicideAttemptPats
FROM #TempSuicideAttemptEnc e
GO

DECLARE @timestamp nvarchar(50)
SET @timestamp = convert(nvarchar(50), GETDATE(), 121)
RAISERROR('%s: Finished suicide attempt', 10, 1, @timestamp) WITH NOWAIT
GO


drop table if exists #TempBipolarDepression_PatientLevelVars;

-- Get all patient-level variables requested by the investigator
SELECT DISTINCT
	p.*,
	adi.ADIUSPercentileRank,
	food_worry.Value AS FoodWorry,
	food_worry.DateKey AS FoodWorryDate,
	alc_drinks_day.NumericValue AS AlcDrinksPerDay,
	alc_drinks_day.DateKey AS AlcDrinksPerDayDate,
	hist_alc_use.Value AS HistoryAlcUse,
	hist_alc_use.DateKey AS HistoryAlcUseDate,
	comm_disease.Value AS CommDiseaseExp,
	comm_disease.DateKey AS CommDiseaseExpDate,
	soc_conn_member.Value AS SocConnMember,
	soc_conn_member.DateKey AS SocConnMemberDate,
	soc_conn_phone.Value AS SocConnPhone,
	soc_conn_phone.DateKey AS SocConnPhoneDate,
	trans_med.Value AS TransportMed,
	trans_med.DateKey AS TransportMedDate,
	cig_pack_years.NumericValue AS CigPackYears,
	cig_pack_years.DateKey AS CigPackYearsDate,
	birth_control.Value AS BirthControl,
	birth_control.DateKey AS BirthControlDate,
	food_scarcity.Value AS FoodScarcity,
	food_scarcity.DateKey AS FoodScarcityDate,
	soc_conn_get_tog.Value AS SocConnGetTog,
	soc_conn_get_tog.DateKey AS SocConnGetTogDate,
	ipv_phys_abuse.Value AS IPVPhysAbuse,
	ipv_phys_abuse.DateKey AS IPVPhysAbuseDate,
	trans_non_med.Value AS TransportNonMed,
	trans_non_med.DateKey AS TransportNonMedDate,
	freq_drug_misuse.NumericValue AS FreqDrugMisuse,
	freq_drug_misuse.DateKey AS FreqDrugMisuseDate,
	cig_packs_day.NumericValue AS CigPacksPerDay,
	cig_packs_day.DateKey AS CigPacksPerDayDate,
	smoking_status.Value AS SmokingStatus,
	smoking_status.DateKey AS SmokingStatusDate,
	phys_activity_dpw.Value AS PhysActivityDPW,
	phys_activity_dpw.DateKey AS PhysActivityDPWDate,
	stress.Value AS Stress,
	stress.DateKey AS StressDate,
	ipv_emotional.Value AS IPVEmotional,
	ipv_emotional.DateKey AS IPVEmotionalDate,
	soc_conn_church.Value AS SocConnChurch,
	soc_conn_church.DateKey AS SocConnChurchDate,
	phys_activity_mps.Value AS PhysActivityMPS,
	phys_activity_mps.DateKey AS PhysActivityMPSDate,
	smokeless_status.Value AS SmokelessStatus,
	smokeless_status.DateKey AS SmokelessStatusDate,
	abused_subst.Value AS AbusedSubstance,
	abused_subst.DateKey AS AbusedSubstanceDate,
	soc_conn_living.Value AS SocConnLiving,
	soc_conn_living.DateKey AS SocConnLivingDate,
	sex_active.Value AS SexuallyActive,
	sex_active.DateKey AS SexuallyActiveDate,
	alc_std_drinks.Value AS AlcStdDrinks,
	alc_std_drinks.DateKey AS AlcStdDrinksDate,
	ipv_fear.Value AS IPVFear,
	ipv_fear.DateKey AS IPVFearDate,
	financial.Value AS Financial,
	financial.DateKey AS FinancialDate,
	tob_use.Value AS TobaccoUse,
	tob_use.DateKey AS TobaccoUseDate,
	travel_hist.Value AS TravelHistory,
	travel_hist.DateKey AS TravelHistoryDate,
	alc_freq.Value AS AlcoholFreq,
	alc_freq.DateKey AS AlcoholFreqDate,
	alc_binge.Value AS AlcoholBinge,
	alc_binge.DateKey AS AlcoholBingeDate,
	ipv_sex_abuse.Value AS IPVSexualAbuse,
	ipv_sex_abuse.DateKey AS IPVSexualAbuseDate,
	subst_use_status.Value AS SubstUseStatus,
	subst_use_status.DateKey AS SubstUseStatusDate,
	soc_conn_meetings.Value AS SocConnMeetings,
	soc_conn_meetings.DateKey AS SocConnMeetingsDate,
	sexual_partner.Value AS SexualPartner,
	sexual_partner.DateKey AS SexualPartnerDate,
	place_lived.NumericValue AS HousingPlaceLived,
	place_lived.DateKey AS HousingPlaceLivedDate,
	homeless.Value AS HousingHomeless,
	homeless.DateKey AS HousingHomelessDate,
	mortgage.Value AS HousingMortgage,
	mortgage.DateKey AS HousingMortgageDate,
	CASE WHEN acute_pain.PatientDurableKey IS NOT NULL THEN 'Y' ELSE 'N' END AS AcutePain,
	CASE WHEN arthropathies.PatientDurableKey IS NOT NULL THEN 'Y' ELSE 'N' END AS Arthropathies,
	CASE WHEN cardiovascular.PatientDurableKey IS NOT NULL THEN 'Y' ELSE 'N' END AS Cardiovascular,
	CASE WHEN chlamydia.PatientDurableKey IS NOT NULL THEN 'Y' ELSE 'N' END AS Chlamydia,
	CASE WHEN chronic_fatigue.PatientDurableKey IS NOT NULL THEN 'Y' ELSE 'N' END AS ChronicFatigue,
	CASE WHEN chronic_pain.PatientDurableKey IS NOT NULL THEN 'Y' ELSE 'N' END AS ChronicPain,
	CASE WHEN covid_19.PatientDurableKey IS NOT NULL THEN 'Y' ELSE 'N' END AS Covid19,
	CASE WHEN dementing.PatientDurableKey IS NOT NULL THEN 'Y' ELSE 'N' END AS Dementing,
	CASE WHEN diabetes.PatientDurableKey IS NOT NULL THEN 'Y' ELSE 'N' END AS Diabetes,
	CASE WHEN gait_impairment.PatientDurableKey IS NOT NULL THEN 'Y' ELSE 'N' END AS GaitImp,
	CASE WHEN gonorrhea.PatientDurableKey IS NOT NULL THEN 'Y' ELSE 'N' END AS Gonorrhea,
	CASE WHEN herpes.PatientDurableKey IS NOT NULL THEN 'Y' ELSE 'N' END AS Herpes,
	CASE WHEN hiv.PatientDurableKey IS NOT NULL THEN 'Y' ELSE 'N' END AS HIV,
	CASE WHEN neoplasms.PatientDurableKey IS NOT NULL THEN 'Y' ELSE 'N' END AS Neoplasms,
	CASE WHEN malnutrition.PatientDurableKey IS NOT NULL THEN 'Y' ELSE 'N' END AS Malnutrition,
	CASE WHEN nervous_sys.PatientDurableKey IS NOT NULL THEN 'Y' ELSE 'N' END AS NervousSys,
	CASE WHEN obesity.PatientDurableKey IS NOT NULL THEN 'Y' ELSE 'N' END AS Obesity,
	CASE WHEN pregnancy_abortive.PatientDurableKey IS NOT NULL THEN 'Y' ELSE 'N' END AS PregnancyAbortive,
	CASE WHEN sleep_apnea.PatientDurableKey IS NOT NULL THEN 'Y' ELSE 'N' END AS SleepApnea,
	CASE WHEN syphilis.PatientDurableKey IS NOT NULL THEN 'Y' ELSE 'N' END AS Syphilis,
	CASE WHEN tbi.PatientDurableKey IS NOT NULL THEN 'Y' ELSE 'N' END AS TBI,
	CASE WHEN weakness.PatientDurableKey IS NOT NULL THEN 'Y' ELSE 'N' END AS Weakness,
	CASE WHEN hospice.PatientDurableKey IS NOT NULL THEN 'Y' ELSE 'N' END AS Hospice,
	CASE WHEN suicideattempt.PatientDurableKey IS NOT NULL THEN 'Y' ELSE 'N' END AS SuicideAttempt
INTO #TempBipolarDepression_PatientLevelVars
FROM #TempBipolarDepressionPats p
LEFT JOIN COSMOS.dbo.PatientAdiDimX adi
	ON p.PatientDurableKey = adi.PatientDurableKey
	AND adi.Version = '3.2'	-- The only version there was, as of 12/27/2024
-- Flowsheet data
LEFT JOIN #TempBipolarDepressionFlowsheetData  food_worry
        ON p.PatientDurableKey = food_worry.PatientDurableKey
        AND food_worry.FlowsheetRowKey = 461
LEFT JOIN #TempBipolarDepressionFlowsheetData  alc_drinks_day
        ON p.PatientDurableKey = alc_drinks_day.PatientDurableKey
        AND alc_drinks_day.FlowsheetRowKey = 1310
LEFT JOIN #TempBipolarDepressionFlowsheetData  hist_alc_use
        ON p.PatientDurableKey = hist_alc_use.PatientDurableKey
        AND hist_alc_use.FlowsheetRowKey = 1888
LEFT JOIN #TempBipolarDepressionFlowsheetData  comm_disease
        ON p.PatientDurableKey = comm_disease.PatientDurableKey
        AND comm_disease.FlowsheetRowKey = 4199
LEFT JOIN #TempBipolarDepressionFlowsheetData  soc_conn_member
        ON p.PatientDurableKey = soc_conn_member.PatientDurableKey
        AND soc_conn_member.FlowsheetRowKey = 4382
LEFT JOIN #TempBipolarDepressionFlowsheetData  soc_conn_phone
        ON p.PatientDurableKey = soc_conn_phone.PatientDurableKey
        AND soc_conn_phone.FlowsheetRowKey = 5549
LEFT JOIN #TempBipolarDepressionFlowsheetData  trans_med
        ON p.PatientDurableKey = trans_med.PatientDurableKey
        AND trans_med.FlowsheetRowKey = 6641
LEFT JOIN #TempBipolarDepressionFlowsheetData  cig_pack_years
        ON p.PatientDurableKey = cig_pack_years.PatientDurableKey
        AND cig_pack_years.FlowsheetRowKey = 6851
LEFT JOIN #TempBipolarDepressionFlowsheetData  birth_control
        ON p.PatientDurableKey = birth_control.PatientDurableKey
        AND birth_control.FlowsheetRowKey = 7542
LEFT JOIN #TempBipolarDepressionFlowsheetData  food_scarcity
        ON p.PatientDurableKey = food_scarcity.PatientDurableKey
        AND food_scarcity.FlowsheetRowKey = 10552
LEFT JOIN #TempBipolarDepressionFlowsheetData  soc_conn_get_tog
        ON p.PatientDurableKey = soc_conn_get_tog.PatientDurableKey
        AND soc_conn_get_tog.FlowsheetRowKey = 11918
LEFT JOIN #TempBipolarDepressionFlowsheetData  ipv_phys_abuse
        ON p.PatientDurableKey = ipv_phys_abuse.PatientDurableKey
        AND ipv_phys_abuse.FlowsheetRowKey = 14738
LEFT JOIN #TempBipolarDepressionFlowsheetData  trans_non_med
        ON p.PatientDurableKey = trans_non_med.PatientDurableKey
        AND trans_non_med.FlowsheetRowKey = 15039
LEFT JOIN #TempBipolarDepressionFlowsheetData  freq_drug_misuse
        ON p.PatientDurableKey = freq_drug_misuse.PatientDurableKey
        AND freq_drug_misuse.FlowsheetRowKey = 16403
LEFT JOIN #TempBipolarDepressionFlowsheetData  cig_packs_day
        ON p.PatientDurableKey = cig_packs_day.PatientDurableKey
        AND cig_packs_day.FlowsheetRowKey = 16585
LEFT JOIN #TempBipolarDepressionFlowsheetData  smoking_status
        ON p.PatientDurableKey = smoking_status.PatientDurableKey
        AND smoking_status.FlowsheetRowKey = 17334
LEFT JOIN #TempBipolarDepressionFlowsheetData  phys_activity_dpw
        ON p.PatientDurableKey = phys_activity_dpw.PatientDurableKey
        AND phys_activity_dpw.FlowsheetRowKey = 17882
LEFT JOIN #TempBipolarDepressionFlowsheetData  stress
        ON p.PatientDurableKey = stress.PatientDurableKey
        AND stress.FlowsheetRowKey = 18000
LEFT JOIN #TempBipolarDepressionFlowsheetData  ipv_emotional
        ON p.PatientDurableKey = ipv_emotional.PatientDurableKey
        AND ipv_emotional.FlowsheetRowKey = 18175
LEFT JOIN #TempBipolarDepressionFlowsheetData  soc_conn_church
        ON p.PatientDurableKey = soc_conn_church.PatientDurableKey
        AND soc_conn_church.FlowsheetRowKey = 20041
LEFT JOIN #TempBipolarDepressionFlowsheetData  phys_activity_mps
        ON p.PatientDurableKey = phys_activity_mps.PatientDurableKey
        AND phys_activity_mps.FlowsheetRowKey = 20465
LEFT JOIN #TempBipolarDepressionFlowsheetData  smokeless_status
        ON p.PatientDurableKey = smokeless_status.PatientDurableKey
        AND smokeless_status.FlowsheetRowKey = 21290
LEFT JOIN #TempBipolarDepressionFlowsheetData  abused_subst
        ON p.PatientDurableKey = abused_subst.PatientDurableKey
        AND abused_subst.FlowsheetRowKey = 22184
LEFT JOIN #TempBipolarDepressionFlowsheetData  soc_conn_living
        ON p.PatientDurableKey = soc_conn_living.PatientDurableKey
        AND soc_conn_living.FlowsheetRowKey = 22629
LEFT JOIN #TempBipolarDepressionFlowsheetData  sex_active
        ON p.PatientDurableKey = sex_active.PatientDurableKey
        AND sex_active.FlowsheetRowKey = 22855
LEFT JOIN #TempBipolarDepressionFlowsheetData  alc_std_drinks
        ON p.PatientDurableKey = alc_std_drinks.PatientDurableKey
        AND alc_std_drinks.FlowsheetRowKey = 23206
LEFT JOIN #TempBipolarDepressionFlowsheetData  ipv_fear
        ON p.PatientDurableKey = ipv_fear.PatientDurableKey
        AND ipv_fear.FlowsheetRowKey = 25468
LEFT JOIN #TempBipolarDepressionFlowsheetData  financial
        ON p.PatientDurableKey = financial.PatientDurableKey
        AND financial.FlowsheetRowKey = 26011
LEFT JOIN #TempBipolarDepressionFlowsheetData  tob_use
        ON p.PatientDurableKey = tob_use.PatientDurableKey
        AND tob_use.FlowsheetRowKey = 26861
LEFT JOIN #TempBipolarDepressionFlowsheetData  travel_hist
        ON p.PatientDurableKey = travel_hist.PatientDurableKey
        AND travel_hist.FlowsheetRowKey = 27138
LEFT JOIN #TempBipolarDepressionFlowsheetData  alc_freq
        ON p.PatientDurableKey = alc_freq.PatientDurableKey
        AND alc_freq.FlowsheetRowKey = 28015
LEFT JOIN #TempBipolarDepressionFlowsheetData  alc_binge
        ON p.PatientDurableKey = alc_binge.PatientDurableKey
        AND alc_binge.FlowsheetRowKey = 28510
LEFT JOIN #TempBipolarDepressionFlowsheetData  ipv_sex_abuse
        ON p.PatientDurableKey = ipv_sex_abuse.PatientDurableKey
        AND ipv_sex_abuse.FlowsheetRowKey = 30044
LEFT JOIN #TempBipolarDepressionFlowsheetData  subst_use_status
        ON p.PatientDurableKey = subst_use_status.PatientDurableKey
        AND subst_use_status.FlowsheetRowKey = 30624
LEFT JOIN #TempBipolarDepressionFlowsheetData  soc_conn_meetings
        ON p.PatientDurableKey = soc_conn_meetings.PatientDurableKey
        AND soc_conn_meetings.FlowsheetRowKey = 30867
LEFT JOIN #TempBipolarDepressionFlowsheetData  sexual_partner
        ON p.PatientDurableKey = sexual_partner.PatientDurableKey
        AND sexual_partner.FlowsheetRowKey = 30944
LEFT JOIN #TempBipolarDepressionFlowsheetData  place_lived
        ON p.PatientDurableKey = place_lived.PatientDurableKey
        AND place_lived.FlowsheetRowKey = 39495
LEFT JOIN #TempBipolarDepressionFlowsheetData  homeless
        ON p.PatientDurableKey = homeless.PatientDurableKey
        AND homeless.FlowsheetRowKey = 39497
LEFT JOIN #TempBipolarDepressionFlowsheetData  mortgage
        ON p.PatientDurableKey = mortgage.PatientDurableKey
        AND mortgage.FlowsheetRowKey = 39499
-- Comorbidities
LEFT JOIN #TempBipolarDepressionComorbidities  acute_pain
        ON p.PatientDurableKey = acute_pain.PatientDurableKey
        AND acute_pain.Category = 'Acute pain'
LEFT JOIN #TempBipolarDepressionComorbidities  arthropathies
        ON p.PatientDurableKey = arthropathies.PatientDurableKey
        AND arthropathies.Category = 'Arthropathies'
LEFT JOIN #TempBipolarDepressionComorbidities  cardiovascular
        ON p.PatientDurableKey = cardiovascular.PatientDurableKey
        AND cardiovascular.Category = 'Cardiovascular'
LEFT JOIN #TempBipolarDepressionComorbidities  chlamydia
        ON p.PatientDurableKey = chlamydia.PatientDurableKey
        AND chlamydia.Category = 'Chlamydia'
LEFT JOIN #TempBipolarDepressionComorbidities  chronic_fatigue
        ON p.PatientDurableKey = chronic_fatigue.PatientDurableKey
        AND chronic_fatigue.Category = 'Chronic fatigue'
LEFT JOIN #TempBipolarDepressionComorbidities  chronic_pain
        ON p.PatientDurableKey = chronic_pain.PatientDurableKey
        AND chronic_pain.Category = 'Chronic pain'
LEFT JOIN #TempBipolarDepressionComorbidities  covid_19
        ON p.PatientDurableKey = covid_19.PatientDurableKey
        AND covid_19.Category = 'COVID-19'
LEFT JOIN #TempBipolarDepressionComorbidities  dementing
        ON p.PatientDurableKey = dementing.PatientDurableKey
        AND dementing.Category = 'Dementing'
LEFT JOIN #TempBipolarDepressionComorbidities  diabetes
        ON p.PatientDurableKey = diabetes.PatientDurableKey
        AND diabetes.Category = 'Diabetes'
LEFT JOIN #TempBipolarDepressionComorbidities  gait_impairment
        ON p.PatientDurableKey = gait_impairment.PatientDurableKey
        AND gait_impairment.Category = 'Gait impairment'
LEFT JOIN #TempBipolarDepressionComorbidities  gonorrhea
        ON p.PatientDurableKey = gonorrhea.PatientDurableKey
        AND gonorrhea.Category = 'Gonorrhea'
LEFT JOIN #TempBipolarDepressionComorbidities  herpes
        ON p.PatientDurableKey = herpes.PatientDurableKey
        AND herpes.Category = 'Herpes'
LEFT JOIN #TempBipolarDepressionComorbidities  hiv
        ON p.PatientDurableKey = hiv.PatientDurableKey
        AND hiv.Category = 'HIV'
LEFT JOIN #TempBipolarDepressionComorbidities  neoplasms
        ON p.PatientDurableKey = neoplasms.PatientDurableKey
        AND neoplasms.Category = 'Malignant and in situ neoplasms'
LEFT JOIN #TempBipolarDepressionComorbidities  malnutrition
        ON p.PatientDurableKey = malnutrition.PatientDurableKey
        AND malnutrition.Category = 'Malnutrition'
LEFT JOIN #TempBipolarDepressionComorbidities  nervous_sys
        ON p.PatientDurableKey = nervous_sys.PatientDurableKey
        AND nervous_sys.Category = 'Nervous system'
LEFT JOIN #TempBipolarDepressionComorbidities  obesity
        ON p.PatientDurableKey = obesity.PatientDurableKey
        AND obesity.Category = 'Obesity'
LEFT JOIN #TempBipolarDepressionComorbidities  pregnancy_abortive
        ON p.PatientDurableKey = pregnancy_abortive.PatientDurableKey
        AND pregnancy_abortive.Category = 'Pregnancy with abortive outcome'
LEFT JOIN #TempBipolarDepressionComorbidities  sleep_apnea
        ON p.PatientDurableKey = sleep_apnea.PatientDurableKey
        AND sleep_apnea.Category = 'Sleep apnea'
LEFT JOIN #TempBipolarDepressionComorbidities  syphilis
        ON p.PatientDurableKey = syphilis.PatientDurableKey
        AND syphilis.Category = 'Syphilis'
LEFT JOIN #TempBipolarDepressionComorbidities  tbi
        ON p.PatientDurableKey = tbi.PatientDurableKey
        AND tbi.Category = 'Traumatic Brain Injury/Concussion'
LEFT JOIN #TempBipolarDepressionComorbidities  weakness
        ON p.PatientDurableKey = weakness.PatientDurableKey
        AND weakness.Category = 'Weakness'
LEFT JOIN #TempBipolarDepressionHospice hospice
	ON p.PatientDurableKey = hospice.PatientDurableKey
LEFT JOIN #TempSuicideAttemptPats suicideattempt
	ON suicideattempt.PatientDurableKey = p.PatientDurableKey
GO
DECLARE @timestamp nvarchar(50)
SET @timestamp = convert(nvarchar(50), GETDATE(), 121)
RAISERROR('%s: Finished patient-level variables', 10, 1, @timestamp) WITH NOWAIT
GO

INSERT INTO PROJECTS.ProjectD52021F.[dbo].[BipolarDepression_PatientLevelVars]
SELECT * FROM #TempBipolarDepression_PatientLevelVars
;

-- Encounter-level conditions
DROP TABLE IF EXISTS #TempBipolarDepressionEncConditions
SELECT DISTINCT
	e.EncounterKey,
	cl.Category
INTO #TempBipolarDepressionEncConditions
FROM dbo.DiagnosisEventFact dx
JOIN dbo.DateDim dxdate
	ON dx.StartDateKey = dxdate.DateKey
	AND dx.Type IN  (N'Billing Final Diagnosis', N'Encounter Diagnosis', N'Billing Admission Diagnosis', N'Discharge Diagnosis', N'Admitting Diagnosis') 
	AND dx.Count = 1
JOIN PROJECTS.ProjectD0F7BC.dbo.MV_SuicideAttempt_ICD10_Conditions_CodeList cl
	ON dx.DiagnosisKey = cl.DiagnosisKey
JOIN #TempBipolarDepressionEnc e
	ON e.PatientDurableKey = dx.PatientDurableKey
	AND dxdate.DateValue >= DATEADD(YEAR, -1, e.EncDate) AND dxdate.DateValue  < e.EncDate
GO
DECLARE @timestamp nvarchar(50)
SET @timestamp = convert(nvarchar(50), GETDATE(), 121)
RAISERROR('%s: Finished encounter-level conditions', 10, 1, @timestamp) WITH NOWAIT
GO

-- Patient reported outcomes
DROP TABLE IF EXISTS #TempBipolarDepressionPRO
SELECT
	EncounterKey,
	NumericResponse,
	ResponseDateKey,
	'PROMIS Pain Intensity 3A' AS Measure
INTO #TempBipolarDepressionPRO
FROM (
SELECT
	e.EncounterKey,
	sa.NumericResponse,
	sa.ResponseDateKey,
	ROW_NUMBER() OVER (PARTITION BY e.EncounterKey ORDER BY sa.ResponseDateKey DESC) AS Line
FROM #TempBipolarDepressionEnc e
JOIN dbo.SurveyAnswerFact sa
	ON e.PatientDurableKey = sa.PatientDurableKey
	AND sa.ResponseDateKey <= e.EncDateKey
	AND sa.NumericResponse IS NOT NULL
	AND sa.Count = 1
	AND sa.SurveyKey IN (105)	-- PROMIS SCALE V1.0-PAIN INTENSITY 3A
	AND sa.SurveyQuestionKey IN (923, 1783)	-- PROMIS Pain Intensity T-Score
) AS q
WHERE Line = 1
;

-- VAS
INSERT INTO #TempBipolarDepressionPRO
SELECT
	EncounterKey,
	NumericResponse,
	ResponseDateKey,
	'VAS' AS Measure
FROM (
SELECT
	e.EncounterKey,
	sa.NumericResponse,
	sa.ResponseDateKey,
	ROW_NUMBER() OVER (PARTITION BY e.EncounterKey ORDER BY sa.ResponseDateKey DESC) AS Line
FROM #TempBipolarDepressionEnc e
JOIN dbo.SurveyAnswerFact sa
	ON e.PatientDurableKey = sa.PatientDurableKey
	AND sa.ResponseDateKey <= e.EncDateKey
	AND sa.NumericResponse IS NOT NULL
	AND sa.Count = 1
	AND sa.SurveyKey IN (33)	-- VISUAL ANALOG SCALE (VAS) FOR PAIN
	AND sa.SurveyQuestionKey IN (330)	-- Visual Analog Scale (VAS) for Pain Score
) AS q
WHERE Line = 1
;

INSERT INTO #TempBipolarDepressionPRO
/*SurveyKey 106 and 390 are the only surveys with answers and 106 does not contain any T-scores
*/
SELECT
	EncounterKey,
	NumericResponse,
	ResponseDateKey,
	'PROMIS Fatigue' AS Measure
FROM (
SELECT
	e.EncounterKey,
	sa.NumericResponse,
	sa.ResponseDateKey,
	ROW_NUMBER() OVER (PARTITION BY e.EncounterKey ORDER BY sa.ResponseDateKey DESC) AS Line
FROM #TempBipolarDepressionEnc e
JOIN dbo.SurveyAnswerFact sa
	ON e.PatientDurableKey = sa.PatientDurableKey
	AND sa.ResponseDateKey <= e.EncDateKey
	AND sa.NumericResponse IS NOT NULL
	AND sa.Count = 1
	AND sa.SurveyKey IN (390)	-- PROMIS PED SHORT FORM V2.0-FATIGUE 10A
	AND sa.SurveyQuestionKey IN (10091)	-- T-score
) AS q
WHERE Line = 1
;

-- Pain Interference
INSERT INTO #TempBipolarDepressionPRO
SELECT
	EncounterKey,
	NumericResponse,
	ResponseDateKey,
	'PROMIS Pain Interference' AS Measure
FROM (
SELECT
	e.EncounterKey,
	sa.NumericResponse,
	sa.ResponseDateKey,
	ROW_NUMBER() OVER (PARTITION BY e.EncounterKey ORDER BY sa.ResponseDateKey DESC) AS Line
FROM #TempBipolarDepressionEnc e
JOIN dbo.SurveyAnswerFact sa
	ON e.PatientDurableKey = sa.PatientDurableKey
	AND sa.ResponseDateKey <= e.EncDateKey
	AND sa.NumericResponse IS NOT NULL
	AND sa.Count = 1
	AND sa.SurveyKey IN (9, 495)	-- PROMIS PED CAT V2.0 - PAIN INTERFERENCE and PROMIS PED CAT V2.0 (SPANISH) - PAIN INTERFERENCE
	AND sa.SurveyQuestionKey IN (755, 10628)	-- T-scores
) AS q1
WHERE Line = 1
GO
DECLARE @timestamp nvarchar(50)
SET @timestamp = convert(nvarchar(50), GETDATE(), 121)
RAISERROR('%s: Finished PRO', 10, 1, @timestamp) WITH NOWAIT
GO

-- Encounter weight
DROP TABLE IF EXISTS #TempBipolarDepressionEncWeight
SELECT
	EncounterKey,
	EncDateKey,
	Weight,
	DateKey AS WeightDate
INTO #TempBipolarDepressionEncWeight
FROM (
SELECT
	e.EncounterKey,
	e.EncDateKey,
	v.Weight,
	v.DateKey,
	ROW_NUMBER() OVER (PARTITION BY e.EncounterKey ORDER BY v.DateKey DESC) AS Line
FROM #TempBipolarDepressionEnc e
JOIN dbo.VitalsFact v
	ON e.PatientDurableKey = v.PatientDurableKey
	AND v.DateKey <= e.EncDateKey
	AND v.Weight IS NOT NULL
	AND v.Count = 1
) AS q
WHERE Line = 1
GO
DECLARE @timestamp nvarchar(50)
SET @timestamp = convert(nvarchar(50), GETDATE(), 121)
RAISERROR('%s: Finished encounter weight', 10, 1, @timestamp) WITH NOWAIT
GO

-- This is the weight we compare the encounter weight against to calculate weight gain and loss: the last recorded weight 1 year before the encounter
DROP TABLE IF EXISTS #TempBipolarDepressionPastWeight
SELECT
	EncounterKey,
	EncDateKey,
	AgeAtEncYears,
	Weight,
	DateKey AS WeightDate
INTO #TempBipolarDepressionPastWeight
FROM (
SELECTRa
	e.EncounterKey,
	e.EncDateKey,
	dd.Years AS AgeAtEncYears,
	v.Weight,
	v.DateKey,
	ROW_NUMBER() OVER (PARTITION BY e.EncounterKey ORDER BY v.DateKey DESC) AS Line
FROM #TempBipolarDepressionEnc e
JOIN dbo.DurationDim dd
	ON e.EncAgeKey = dd.DurationKey
JOIN dbo.VitalsFact v
	ON e.PatientDurableKey = v.PatientDurableKey
	AND v.DateKey BETWEEN CASE WHEN dd.Years >= 16 THEN 19000101 ELSE e.EncDateKey - 20000 END AND e.EncDateKey - 10000	-- When the patient is older than 16 at encounter date, we can look back as much as we need to (limit is 1900-01-01), but when the age is 16 or younger, we can only look back 2 years before encounter date
	AND v.Weight IS NOT NULL
	AND v.Count = 1
) q
WHERE
	Line = 1
GO
DECLARE @timestamp nvarchar(50)
SET @timestamp = convert(nvarchar(50), GETDATE(), 121)
RAISERROR('%s: Finished encounter past weight', 10, 1, @timestamp) WITH NOWAIT
GO

-- Encounter height
DROP TABLE IF EXISTS #TempBipolarDepressionEncHeight
SELECT
	EncounterKey,
	EncDateKey,
	Height,
	DateKey AS HeightDate
INTO #TempBipolarDepressionEncHeight
FROM (
SELECT
	e.EncounterKey,
	e.EncDateKey,
	v.Height,
	v.DateKey,
	ROW_NUMBER() OVER (PARTITION BY e.EncounterKey ORDER BY v.DateKey DESC) AS Line
FROM #TempBipolarDepressionEnc e
JOIN dbo.VitalsFact v
	ON e.PatientDurableKey = v.PatientDurableKey
	AND v.DateKey <= e.EncDateKey
	AND v.Height IS NOT NULL
	AND v.Count = 1
) AS q
WHERE Line = 1
GO
DECLARE @timestamp nvarchar(50)
SET @timestamp = convert(nvarchar(50), GETDATE(), 121)
RAISERROR('%s: Finished encounter height', 10, 1, @timestamp) WITH NOWAIT
GO

-- Encounter BMI
DROP TABLE IF EXISTS #TempBipolarDepressionEncBMI
SELECT
	EncounterKey,
	EncDateKey,
	BodyMassIndex AS BMI,
	DateKey AS BMIDate
INTO #TempBipolarDepressionEncBMI
FROM (
SELECT
	e.EncounterKey,
	e.EncDateKey,
	v.BodyMassIndex,
	v.DateKey,
	ROW_NUMBER() OVER (PARTITION BY e.EncounterKey ORDER BY v.DateKey DESC) AS Line
FROM #TempBipolarDepressionEnc e
JOIN dbo.VitalsFact v
	ON e.PatientDurableKey = v.PatientDurableKey
	AND v.DateKey <= e.EncDateKey
	AND v.BodyMassIndex IS NOT NULL
	AND v.Count = 1
) AS q
WHERE Line = 1
GO
DECLARE @timestamp nvarchar(50)
SET @timestamp = convert(nvarchar(50), GETDATE(), 121)
RAISERROR('%s: Finished encounter BMI', 10, 1, @timestamp) WITH NOWAIT
GO

-- Health Insurance Loss
-- If MarkedSelfPay was "yes" at the encounter in question and  "no" at the last encounter before that, then we assume there was a loss of health insurance
DROP TABLE IF EXISTS #TempBipolarDepressionHealthInsLoss
SELECT EncounterKey
INTO #TempBipolarDepressionHealthInsLoss
FROM (
SELECT
	e.EncounterKey,
	e.MarkedSelfPay AS CurrentSelfPay,
	ef.MarkedSelfPay AS PastSelfPay,
	ROW_NUMBER() OVER (PARTITION BY e.EncounterKey ORDER BY ef.DateKey DESC) AS Line
FROM #TempBipolarDepressionEnc e
JOIN dbo.EncounterFact ef
	ON e.PatientDurableKey = ef.PatientDurableKey
	AND ef.DateKey < e.EncDateKey
	AND ef.Count = 1
) q
WHERE
	CurrentSelfPay = 1
	AND PastSelfPay = 0
	AND Line = 1
GO
DECLARE @timestamp nvarchar(50)
SET @timestamp = convert(nvarchar(50), GETDATE(), 121)
RAISERROR('%s: Finished health insurance loss', 10, 1, @timestamp) WITH NOWAIT
GO

-- Psychiatry visits
DROP TABLE IF EXISTS #TempBipolarDepressionPsychiatry
SELECT 
	e.EncounterKey,
	COUNT(DISTINCT ef.EncounterKey) AS VisitCount
INTO #TempBipolarDepressionPsychiatry
FROM #TempBipolarDepressionEnc e
JOIN dbo.EncounterFact ef
	ON e.PatientDurableKey = ef.PatientDurableKey
	AND (ef.IsHospitalOutpatientVisit = 1 OR ef.IsOutpatientFaceToFaceVisit = 1)
	AND ef.Date >=  DATEADD(m, -3, e.EncDate) AND ef.Date < e.EncDate
	AND ef.Count = 1
JOIN dbo.ProviderDim prov
	ON ef.ProviderDurableKey = prov.DurableKey
WHERE
	prov.PrimarySpecialty IN (
'Psychiatric/Mental Health',
'Psychiatric/Mental Health, Adult',
'Psychiatric/Mental Health, Child & Adolescent',
'Psychiatric/Mental Health, Community',
'Psychiatry',
'Pediatric Psychiatry',
'Addiction Psychiatry',
'Child and Adolescent Psychiatry',
'Neuropsychiatry',
'Geriatric Psychiatry',
'Psychiatric/Mental Health'
)
GROUP BY
	e.EncounterKey
GO
DECLARE @timestamp nvarchar(50)
SET @timestamp = convert(nvarchar(50), GETDATE(), 121)
RAISERROR('%s: Finished psychiatry visits', 10, 1, @timestamp) WITH NOWAIT
GO

-- Psychology visits
DROP TABLE IF EXISTS #TempBipolarDepressionPsychology
SELECT 
	e.EncounterKey,
	COUNT(DISTINCT ef.EncounterKey) AS VisitCount
INTO #TempBipolarDepressionPsychology
FROM #TempBipolarDepressionEnc e
JOIN dbo.EncounterFact ef
	ON e.PatientDurableKey = ef.PatientDurableKey
	AND (ef.IsHospitalOutpatientVisit = 1 OR ef.IsOutpatientFaceToFaceVisit = 1)
	AND ef.Date >=  DATEADD(m, -3, e.EncDate) AND ef.Date < e.EncDate
	AND ef.Count = 1
JOIN dbo.ProviderDim prov
	ON ef.ProviderDurableKey = prov.DurableKey
WHERE
	prov.PrimarySpecialty IN (
'Psychology', 'Pediatric Psychology', 'Neuropsychology'
)
GROUP BY
	e.EncounterKey
GO
DECLARE @timestamp nvarchar(50)
SET @timestamp = convert(nvarchar(50), GETDATE(), 121)
RAISERROR('%s: Finished psychology visits', 10, 1, @timestamp) WITH NOWAIT
GO

-- Outpatient primary ICD-10 codes in the 3 months before the encounter
DROP TABLE IF EXISTS #TempBipolarDepressionVisitDxPrimary
SELECT
	EncounterKey,
	STRING_AGG(ICD10Code, ',')  WITHIN GROUP (ORDER BY ICD10Code) AS Dx
INTO #TempBipolarDepressionVisitDxPrimary
FROM (
SELECT DISTINCT 
	e.EncounterKey,
	dtd.Value AS ICD10Code
FROM #TempBipolarDepressionEnc e
JOIN dbo.EncounterFact ef
	ON e.PatientDurableKey = ef.PatientDurableKey
	AND (ef.IsHospitalOutpatientVisit = 1 OR ef.IsOutpatientFaceToFaceVisit = 1)
	AND ef.Date >=  DATEADD(m, -3, e.EncDate) AND ef.Date < e.EncDate
	AND ef.Count = 1
	AND ef.IsEdVisit = 0
JOIN dbo.DiagnosisEventFact def
	ON ef.EncounterKey  = def.EncounterKey
	AND def.Type IN  (N'Billing Final Diagnosis', N'Encounter Diagnosis', N'Billing Admission Diagnosis', N'Discharge Diagnosis', N'Admitting Diagnosis') 
	AND def.IsPrimary = 1
JOIN dbo.DiagnosisTerminologyDim dtd
	ON def.DiagnosisKey = dtd.DiagnosisKey
	AND dtd.Type = 'ICD-10-CM'
) q
GROUP BY
	EncounterKey
GO
DECLARE @timestamp nvarchar(50)
SET @timestamp = convert(nvarchar(50), GETDATE(), 121)
RAISERROR('%s: Finished outpatient primary ICD-10 codes', 10, 1, @timestamp) WITH NOWAIT
GO

-- Outpatient non-primary ICD-10 codes in the 3 months before the encounter
DROP TABLE IF EXISTS #TempBipolarDepressionVisitDxNonPrimary
SELECT
	EncounterKey,
	STRING_AGG(ICD10Code, ',')  WITHIN GROUP (ORDER BY ICD10Code) AS Dx
INTO #TempBipolarDepressionVisitDxNonPrimary
FROM (
SELECT DISTINCT 
	e.EncounterKey,
	dtd.Value AS ICD10Code
FROM #TempBipolarDepressionEnc e
JOIN dbo.EncounterFact ef
	ON e.PatientDurableKey = ef.PatientDurableKey
	AND (ef.IsHospitalOutpatientVisit = 1 OR ef.IsOutpatientFaceToFaceVisit = 1)
	AND ef.Date >=  DATEADD(m, -3, e.EncDate) AND ef.Date < e.EncDate
	AND ef.Count = 1
	AND ef.IsEdVisit = 0
JOIN dbo.DiagnosisEventFact def
	ON ef.EncounterKey  = def.EncounterKey
	AND def.Type IN  (N'Billing Final Diagnosis', N'Encounter Diagnosis', N'Billing Admission Diagnosis', N'Discharge Diagnosis', N'Admitting Diagnosis') 
	AND ISNULL(def.IsPrimary, 0) = 0
JOIN dbo.DiagnosisTerminologyDim dtd
	ON def.DiagnosisKey = dtd.DiagnosisKey
	AND dtd.Type = 'ICD-10-CM'
) q
GROUP BY
	EncounterKey
GO
DECLARE @timestamp nvarchar(50)
SET @timestamp = convert(nvarchar(50), GETDATE(), 121)
RAISERROR('%s: Finished outpatient non-primary ICD-10 codes', 10, 1, @timestamp) WITH NOWAIT
GO

-- History of suicide attempt
DROP TABLE IF EXISTS #TempBipolarDepressionHxSuicideAttempt
SELECT DISTINCT p.PatientDurableKey
INTO #TempBipolarDepressionHxSuicideAttempt
FROM dbo.DiagnosisEventFact def
JOIN #TempBipolarDepressionPats p
	ON def.PatientDurableKey = p.PatientDurableKey
WHERE 
	def.Type IN  (N'Billing Final Diagnosis', N'Encounter Diagnosis', N'Problem List', N'Billing Admission Diagnosis', N'Discharge Diagnosis', N'Admitting Diagnosis') 
	AND def.DiagnosisKey IN (
154727,		--	H/O: suicide attempt
602189,		--	History of attempted suicide
1033219,	--	History of suicide attempt
65298,		--	Hx of suicide attempt
368785,		--	H/O: attempted suicide
129584		--	H/O suicide attempt
)
GO
DECLARE @timestamp nvarchar(50)
SET @timestamp = convert(nvarchar(50), GETDATE(), 121)
RAISERROR('%s: Finished history of suicide attempt', 10, 1, @timestamp) WITH NOWAIT
GO

-- Medications
-- Opioids (includes cough medicine)
DROP TABLE IF EXISTS #TempBipolarDepressionOpioids
SELECT DISTINCT e.EncounterKey
INTO #TempBipolarDepressionOpioids
FROM dbo.MedicationOrderComponentFact moc
JOIN dbo.EncounterFact ef
	ON moc.EncounterKey = ef.EncounterKey
	AND moc.Count = 1
	AND ef.Count = 1
JOIN dbo.MedicationSetDim ms
	ON moc.MedicationKey = ms.MedicationKey	
	AND ms.ValueSetEpicId IN ('89110959', '9001700014')
JOIN #TempBipolarDepressionEnc e
	ON moc.PatientDurableKey = e.PatientDurableKey
	AND ef.Date >= DATEADD(m, -3, e.EncDate) AND ef.Date < e.EncDate
GO

-- SELECTIVE SEROTONIN REUPTAKE INHIBITOR (SSRI)
DROP TABLE IF EXISTS #TempBipolarDepressionSSRI
SELECT DISTINCT e.EncounterKey
INTO #TempBipolarDepressionSSRI
FROM dbo.MedicationOrderComponentFact moc
JOIN dbo.EncounterFact ef
	ON moc.EncounterKey = ef.EncounterKey
	AND moc.Count = 1
	AND ef.Count = 1
JOIN dbo.MedicationSetDim ms
	ON moc.MedicationKey = ms.MedicationKey	
	AND ms.ValueSetEpicId IN ('891311', '899898', '8991212')	-- SSRI
JOIN dbo.MedicationDim m
	ON ms.MedicationKey = m.MedicationKey
	AND m.TherapeuticClass <> 'MUSCLE RELAXANTS'	-- Muscle relaxants don't seem to be SSRIs
JOIN #TempBipolarDepressionEnc e
	ON moc.PatientDurableKey = e.PatientDurableKey
	AND ef.Date >= DATEADD(m, -3, e.EncDate) AND ef.Date < e.EncDate
GO

-- Antipsychotics
DROP TABLE IF EXISTS #TempBipolarDepressionAntipsychotics
SELECT DISTINCT e.EncounterKey
INTO #TempBipolarDepressionAntipsychotics
FROM dbo.MedicationOrderComponentFact moc
JOIN dbo.EncounterFact ef
	ON moc.EncounterKey = ef.EncounterKey
	AND moc.Count = 1
	AND ef.Count = 1
JOIN dbo.MedicationDim m
	ON moc.MedicationKey = m.MedicationKey
	AND (m.PharmaceuticalClass LIKE '%antipsych%' OR m.TherapeuticClass LIKE '%antipsych%' OR m.PharmaceuticalSubClass LIKE '%antipsych%')
	AND m.MedicationKey <> 168597	-- This one is a Zyrtec Allergy PO with a generic name of haloperidol (an antipsychotic); does not make sense
JOIN #TempBipolarDepressionEnc e
	ON moc.PatientDurableKey = e.PatientDurableKey
	AND ef.Date >= DATEADD(m, -3, e.EncDate) AND ef.Date < e.EncDate
GO

-- Lithium
DROP TABLE IF EXISTS #TempBipolarDepressionLithium
SELECT DISTINCT e.EncounterKey
INTO #TempBipolarDepressionLithium
FROM dbo.MedicationOrderComponentFact moc
JOIN dbo.EncounterFact ef
	ON moc.EncounterKey = ef.EncounterKey
	AND moc.Count = 1
	AND ef.Count = 1
JOIN dbo.MedicationSetDim ms
	ON moc.MedicationKey = ms.MedicationKey	
	AND ms.ValueSetEpicId IN ('8991823')	-- Lithium
JOIN #TempBipolarDepressionEnc e
	ON moc.PatientDurableKey = e.PatientDurableKey
	AND ef.Date >= DATEADD(m, -3, e.EncDate) AND ef.Date < e.EncDate
GO

-- Anticonvulsants
DROP TABLE IF EXISTS #TempBipolarDepressionAnticonvulsants
SELECT DISTINCT e.EncounterKey
INTO #TempBipolarDepressionAnticonvulsants
FROM dbo.MedicationOrderComponentFact moc
JOIN dbo.EncounterFact ef
	ON moc.EncounterKey = ef.EncounterKey
	AND moc.Count = 1
	AND ef.Count = 1
JOIN dbo.MedicationSetDim ms
	ON moc.MedicationKey = ms.MedicationKey	
	AND ms.ValueSetEpicId IN ('899235')	-- Anticonvulsants
JOIN #TempBipolarDepressionEnc e
	ON moc.PatientDurableKey = e.PatientDurableKey
	AND ef.Date >= DATEADD(m, -3, e.EncDate) AND ef.Date < e.EncDate
GO

-- Benzos
DROP TABLE IF EXISTS #TempBipolarDepressionBenzos
SELECT DISTINCT e.EncounterKey
INTO #TempBipolarDepressionBenzos
FROM dbo.MedicationOrderComponentFact moc
JOIN dbo.EncounterFact ef
	ON moc.EncounterKey = ef.EncounterKey
	AND moc.Count = 1
	AND ef.Count = 1
JOIN dbo.MedicationSetDim ms
	ON moc.MedicationKey = ms.MedicationKey	
	AND ms.ValueSetEpicId IN ('8993555','9001700017')	-- Benzodiazepines
JOIN #TempBipolarDepressionEnc e
	ON moc.PatientDurableKey = e.PatientDurableKey
	AND ef.Date >= DATEADD(m, -3, e.EncDate) AND ef.Date < e.EncDate
GO

-- Norepinephrine reuptake inhibitor
DROP TABLE IF EXISTS #TempBipolarDepressionNRI
SELECT DISTINCT e.EncounterKey
INTO #TempBipolarDepressionNRI
FROM dbo.MedicationOrderComponentFact moc
JOIN dbo.EncounterFact ef
	ON moc.EncounterKey = ef.EncounterKey
	AND moc.Count = 1
	AND ef.Count = 1
JOIN dbo.MedicationDim m
	ON moc.MedicationKey = m.MedicationKey
	AND (m.PharmaceuticalClass LIKE '%Norepinephrine%' OR m.TherapeuticClass LIKE '%Norepinephrine%' OR m.PharmaceuticalSubClass LIKE '%Norepinephrine%')
JOIN #TempBipolarDepressionEnc e
	ON moc.PatientDurableKey = e.PatientDurableKey
	AND ef.Date >= DATEADD(m, -3, e.EncDate) AND ef.Date < e.EncDate
GO

-- ADHD Stimulants
DROP TABLE IF EXISTS #TempBipolarDepressionADHD
SELECT DISTINCT e.EncounterKey
INTO #TempBipolarDepressionADHD
FROM dbo.MedicationOrderComponentFact moc
JOIN dbo.EncounterFact ef
	ON moc.EncounterKey = ef.EncounterKey
	AND moc.Count = 1
	AND ef.Count = 1
JOIN dbo.MedicationSetDim ms
	ON moc.MedicationKey = ms.MedicationKey	
	AND ms.ValueSetEpicId IN ('89110981')	-- ADHD Stimulants
JOIN #TempBipolarDepressionEnc e
	ON moc.PatientDurableKey = e.PatientDurableKey
	AND ef.Date >= DATEADD(m, -3, e.EncDate) AND ef.Date < e.EncDate
GO
DECLARE @timestamp nvarchar(50)
SET @timestamp = convert(nvarchar(50), GETDATE(), 121)
RAISERROR('%s: Finished medications', 10, 1, @timestamp) WITH NOWAIT
GO

-- Risk scores
DROP TABLE IF EXISTS #TempBipolarDepressionRiskScores
SELECT
	EncounterKey,
	RiskScoreRuleKey,
	CalculationDateKey,
    Score,
	ScoreLevel
INTO #TempBipolarDepressionRiskScores
FROM (	
SELECT DISTINCT
	e.EncounterKey,
	rsf.RiskScoreRuleKey,
	rsf.CalculationDateKey,
    rsf.Score,
	rsf.ScoreLevel,
	ROW_NUMBER() OVER (PARTITION BY e.EncounterKey, rsf.RiskScoreRuleKey ORDER BY rsf.CalculationInstant DESC) AS Line
FROM [dbo].[RiskScoreFact] rsf
JOIN #TempBipolarDepressionEnc e
	ON  rsf.PatientDurableKey = e.PatientDurableKey
	AND rsf.CalculationDateKey <= e.EncDateKey
) q
WHERE
	Line = 1
GO
DECLARE @timestamp nvarchar(50)
SET @timestamp = convert(nvarchar(50), GETDATE(), 121)
RAISERROR('%s: Finished risk scores', 10, 1, @timestamp) WITH NOWAIT
GO

-- ED Visits counts
DROP TABLE IF EXISTS #TempBipolarDepressionEDVisits
SELECT
	e.EncounterKey,
	COUNT(DISTINCT ef.EncounterKey) AS EDVisitCount
INTO #TempBipolarDepressionEDVisits
FROM #TempBipolarDepressionEnc e
JOIN dbo.EncounterFact ef
	ON e.PatientDurableKey = ef.PatientDurableKey
	AND ef.Date >=  DATEADD(m, -3, e.EncDate) AND ef.Date < e.EncDate
	AND ef.Count = 1
	AND ef.IsEdVisit = 1
GROUP BY
	e.EncounterKey
GO
DECLARE @timestamp nvarchar(50)
SET @timestamp = convert(nvarchar(50), GETDATE(), 121)
RAISERROR('%s: Finished ED visit counts', 10, 1, @timestamp) WITH NOWAIT
GO

-- ED Visits primary dx
DROP TABLE IF EXISTS #TempBipolarDepressionEDDxPrimary
SELECT
	EncounterKey,
	STRING_AGG(ICD10Code, ',')  WITHIN GROUP (ORDER BY ICD10Code) AS EDPrimaryDx
INTO #TempBipolarDepressionEDDxPrimary
FROM (
SELECT DISTINCT 
	e.EncounterKey,
	dtd.Value AS ICD10Code
FROM #TempBipolarDepressionEnc e
JOIN dbo.EncounterFact ef
	ON e.PatientDurableKey = ef.PatientDurableKey
	AND ef.Date >=  DATEADD(m, -3, e.EncDate) AND ef.Date < e.EncDate
	AND ef.Count = 1
	AND ef.IsEdVisit = 1
JOIN dbo.DiagnosisEventFact def
	ON ef.EncounterKey  = def.EncounterKey
	AND def.Type IN  (N'Billing Final Diagnosis', N'Encounter Diagnosis', N'Billing Admission Diagnosis', N'Discharge Diagnosis', N'Admitting Diagnosis') 
	AND def.IsPrimary = 1
JOIN dbo.DiagnosisTerminologyDim dtd
	ON def.DiagnosisKey = dtd.DiagnosisKey
	AND dtd.Type = 'ICD-10-CM'
) q
GROUP BY
	EncounterKey
GO
DECLARE @timestamp nvarchar(50)
SET @timestamp = convert(nvarchar(50), GETDATE(), 121)
RAISERROR('%s: Finished ED visits primary dx', 10, 1, @timestamp) WITH NOWAIT
GO

-- Prescription medications in the last year before the encounter
DROP TABLE IF EXISTS #TempBipolarDepressionPrescripMedCount
SELECT
	e.EncounterKey,
	COUNT(DISTINCT MedicationKey) AS PresMedCount
INTO #TempBipolarDepressionPrescripMedCount
FROM dbo.MedicationOrderComponentFact moc
JOIN dbo.EncounterFact ef
	ON moc.EncounterKey = ef.EncounterKey
	AND moc.Count = 1
	AND ef.Count = 1
	AND moc.Type_X = 'Prescription'
JOIN #TempBipolarDepressionEnc e
	ON moc.PatientDurableKey = e.PatientDurableKey
	AND ef.Date >= DATEADD(YEAR, -1, e.EncDate) AND ef.Date < e.EncDate
GROUP BY
	e.EncounterKey
GO
DECLARE @timestamp nvarchar(50)
SET @timestamp = convert(nvarchar(50), GETDATE(), 121)
RAISERROR('%s: Finished prescription medications', 10, 1, @timestamp) WITH NOWAIT
GO

-- Number of procedures in the last year before the encounter
-- Includes procedures such as "COLLECTION VENOUS BLOOD VENIPUNCTURE 36415"
DROP TABLE IF EXISTS #TempBipolarDepressionPxCount
SELECT
	e.EncounterKey,
	COUNT(DISTINCT pe.ProcedureEventKey) AS PxCount
INTO #TempBipolarDepressionPxCount
FROM dbo.ProcedureEventFact pe
JOIN dbo.DateDim d
	ON pe.ProcedureStartDateKey = d.DateKey
	AND pe.Count = 1
JOIN dbo.ProcedureDim pr
	ON pe.ProcedureDurableKey = pr.DurableKey
	AND pr.CptCode BETWEEN '10000' AND '69999'
JOIN #TempBipolarDepressionEnc e
	ON pe.PatientDurableKey = e.PatientDurableKey
	AND d.DateValue >=  DATEADD(year, -1, e.EncDate) AND d.DateValue < e.EncDate
GROUP BY
	e.EncounterKey
GO
DECLARE @timestamp nvarchar(50)
SET @timestamp = convert(nvarchar(50), GETDATE(), 121)
RAISERROR('%s: Finished procedures', 10, 1, @timestamp) WITH NOWAIT
GO

-- Labs
DROP TABLE IF EXISTS #TempBipolarDepressionLabs
SELECT
	'Pulmonary function test' AS LabType,
	e.EncounterKey,
	MAX(DATEPART(YEAR, lc.ResultInstant)) AS MaxYear
INTO #TempBipolarDepressionLabs
FROM [dbo].[LabComponentResultFact] lc
JOIN #TempBipolarDepressionEnc e
	ON lc.PatientDurableKey = e.PatientDurableKey
	AND lc.Count = 1
	AND lc.IsBlankOrUnsuccessfulAttempt = 0
	AND lc.ResultInstant < e.EncDate
	AND lc.LabComponentKey IN (62104, 53488, 52831)	-- Pulmonary function test
GROUP BY
	e.EncounterKey
GO

INSERT INTO #TempBipolarDepressionLabs
SELECT
	'Fasting glucose' AS LabType,
	e.EncounterKey,
	MAX(DATEPART(YEAR, lc.ResultInstant)) AS MaxYear
FROM [dbo].[LabComponentResultFact] lc
JOIN #TempBipolarDepressionEnc e
	ON lc.PatientDurableKey = e.PatientDurableKey
	AND lc.Count = 1
	AND lc.IsBlankOrUnsuccessfulAttempt = 0
	AND lc.ResultInstant < e.EncDate
	AND lc.LabComponentKey IN (1234,1499,1514,1518,23058,56241,63136,68557,25042,79421,4416,98247,105587,85788,29730,20081,47314,88701,68875,1823)	-- Fasting glucose
GROUP BY
	e.EncounterKey
GO

INSERT INTO #TempBipolarDepressionLabs
SELECT
	'Spot glucose' AS LabType,
	e.EncounterKey,
	MAX(DATEPART(YEAR, lc.ResultInstant)) AS MaxYear
FROM [dbo].[LabComponentResultFact] lc
JOIN #TempBipolarDepressionEnc e
	ON lc.PatientDurableKey = e.PatientDurableKey
	AND lc.Count = 1
	AND lc.IsBlankOrUnsuccessfulAttempt = 0
	AND lc.ResultInstant < e.EncDate
JOIN dbo.LabComponentDim l
	ON lc.LabComponentKey = l.LabComponentKey
	AND CommonName like 'glucose%' AND CommonName NOT LIKE '%fast%' AND CommonName NOT LIKE '%SOMATOTROPIN%' AND CommonName NOT LIKE '%presence%' AND CommonName NOT LIKE '%creatinine%' AND CommonName NOT LIKE '%DEHYDROGENASE%' AND CommonName NOT LIKE '%presence%'
GROUP BY
	e.EncounterKey
GO

INSERT INTO #TempBipolarDepressionLabs
SELECT
	'Lipid levels' AS LabType,
	e.EncounterKey,
	MAX(DATEPART(YEAR, lc.ResultInstant)) AS MaxYear
FROM [dbo].[LabComponentResultFact] lc
JOIN #TempBipolarDepressionEnc e
	ON lc.PatientDurableKey = e.PatientDurableKey
	AND lc.Count = 1
	AND lc.IsBlankOrUnsuccessfulAttempt = 0
	AND lc.ResultInstant < e.EncDate
JOIN dbo.LabComponentDim l
	ON lc.LabComponentKey = l.LabComponentKey
	AND (CommonName LIKE '%ldl%' OR CommonName LIKE '%hdl%' OR CommonName LIKE '%Cholesterol%' OR CommonName LIKE '%Triglycerides%')
GROUP BY
	e.EncounterKey
GO

INSERT INTO #TempBipolarDepressionLabs
SELECT
	'CRP' AS LabType,
	e.EncounterKey,
	MAX(DATEPART(YEAR, lc.ResultInstant)) AS MaxYear
FROM [dbo].[LabComponentResultFact] lc
JOIN #TempBipolarDepressionEnc e
	ON lc.PatientDurableKey = e.PatientDurableKey
	AND lc.Count = 1
	AND lc.IsBlankOrUnsuccessfulAttempt = 0
	AND lc.ResultInstant < e.EncDate
	AND lc.LabComponentKey IN (7623,1364,2178,100568,3907,3299,12471,89953,849,5022,34661,27010,109769)	-- CRP C-reactive protein
GROUP BY
	e.EncounterKey
GO

INSERT INTO #TempBipolarDepressionLabs
SELECT
	'Cortisol' AS LabType,
	e.EncounterKey,
	MAX(DATEPART(YEAR, lc.ResultInstant)) AS MaxYear
FROM [dbo].[LabComponentResultFact] lc
JOIN #TempBipolarDepressionEnc e
	ON lc.PatientDurableKey = e.PatientDurableKey
	AND lc.Count = 1
	AND lc.IsBlankOrUnsuccessfulAttempt = 0
	AND lc.ResultInstant < e.EncDate
JOIN LabComponentDim l
	ON lc.LabComponentKey = l.LabComponentKey
	AND CommonName LIKE '%cortisol%'
GROUP BY
	e.EncounterKey
GO

INSERT INTO #TempBipolarDepressionLabs
SELECT
	'Thyroid' AS LabType,
	e.EncounterKey,
	MAX(DATEPART(YEAR, lc.ResultInstant)) AS MaxYear
FROM [dbo].[LabComponentResultFact] lc
JOIN #TempBipolarDepressionEnc e
	ON lc.PatientDurableKey = e.PatientDurableKey
	AND lc.Count = 1
	AND lc.IsBlankOrUnsuccessfulAttempt = 0
	AND lc.ResultInstant < e.EncDate
JOIN dbo.LabComponentDim l
	ON lc.LabComponentKey = l.LabComponentKey
	AND (CommonName LIKE '%thyroid%' OR CommonName LIKE '%tsh%' OR CommonName LIKE '%t3%' OR CommonName LIKE '%t4%') AND CommonName NOT LIKE '%helper%'
GROUP BY
	e.EncounterKey
GO

DECLARE @timestamp nvarchar(50)
SET @timestamp = convert(nvarchar(50), GETDATE(), 121)
RAISERROR('%s: Finished labs', 10, 1, @timestamp) WITH NOWAIT
GO

-- new requested variable: current suicide attempt: Y/N variable whether the encounter in question was also coded with suicide attempt
DROP TABLE IF EXISTS #TempCurrentSuicideAttempt
SELECT DISTINCT
	dx.EncounterKey
INTO #TempCurrentSuicideAttempt
FROM dbo.DiagnosisEventFact dx
JOIN PROJECTS.ProjectD52021F.[dbo].[BipolarDepression_EncLevelVars] e
	ON dx.EncounterKey = e.EncounterKey
JOIN PROJECTS.ProjectD0F7BC.dbo.MV_SuicideAttempt_ICD10_CodeList_v2 cl
	ON dx.DiagnosisKey = cl.DiagnosisKey
	AND dx.Type IN  (N'Billing Final Diagnosis', N'Encounter Diagnosis', N'Billing Admission Diagnosis', N'Discharge Diagnosis', N'Admitting Diagnosis')
	AND dx.Count = 1
GO
DECLARE @timestamp nvarchar(50)
SET @timestamp = convert(nvarchar(50), GETDATE(), 121)
RAISERROR('%s: Finished current suicide attempt', 10, 1, @timestamp) WITH NOWAIT
GO

INSERT INTO PROJECTS.ProjectD52021F.[dbo].[BipolarDepression_EncLevelVars_v2]
SELECT
	e1.*,
	CASE WHEN csa.EncounterKey IS NOT NULL THEN 'Y' ELSE 'N' END AS CurrentSuicideAttempt
FROM PROJECTS.ProjectD52021F.[dbo].[BipolarDepression_EncLevelVars] e1
LEFT JOIN #TempCurrentSuicideAttempt csa
	ON e1.EncounterKey = csa.EncounterKey
GO

-- Get encounter-level variables
DROP TABLE IF EXISTS #BipolarDepression_EncLevelVars
SELECT DISTINCT
	e.EncounterKey,
	e.PatientDurableKey,
	e.EncDate,
	e.EncDateKey,
	age.Days AS AgeInDays,
	age.Weeks AS AgeInWeeks,
	age.Months AS AgeInMonths,
	age.Years AS AgeInYears,
	ew.Weight AS EncounterWeightLbs,
	CASE WHEN ew.Weight - pw.Weight > 10 AND ew.Weight - pw.Weight <= 20 THEN 'Y' ELSE 'N' END AS WeightGain10_20lbs,
	CASE WHEN ew.Weight - pw.Weight > 20 THEN 'Y' ELSE 'N' END AS WeightGain20pluslbs,
	CASE WHEN pw.Weight - ew.Weight > 10 AND pw.Weight - ew.Weight <= 20 THEN 'Y' ELSE 'N' END AS WeightLoss10_20lbs,
	CASE WHEN pw.Weight - ew.Weight > 20 THEN 'Y' ELSE 'N' END AS WeightLoss20pluslbs,
	eh.Height AS EncounterHeightInches,
	eb.BMI AS EncounterBMI,
	CASE WHEN hil.EncounterKey IS NOT NULL THEN 'Y' ELSE 'N' END AS HealthInsLoss,
	CASE WHEN e.PrimaryCoverageFinancialClass_X = 'Medicare' THEN 'Y' ELSE 'N' END AS,
	CASE WHEN e.PrimaryCoverageFinancialClass_X = 'Medicaid' THEN 'Y' ELSE 'N' END AS MedicaidYN,
	CASE WHEN e.MarkedSelfPay = 1 THEN 'Y' ELSE 'N' END AS SelfPayYN,
	CASE WHEN e.PrimaryCoverageFinancialClass_X = 'Miscellaneous/Other' THEN 'Y' ELSE 'N' END AS MiscOtherYN,
	ISNULL(psychiatry.VisitCount, 0) AS PsychiatryVisitCount,
	ISNULL(psychology.VisitCount, 0) AS PsychologyVisitCount,
	dxprimary.Dx AS OutpatPrimaryDx,
	dxnonprimary.Dx AS OutpatNonPrimaryDx,
	CASE WHEN hxsa.PatientDurableKey IS NOT NULL OR priorsaenc.EncounterKey IS NOT NULL THEN 'Y' ELSE 'N' END AS HxSuicideAttempt,
	CASE WHEN opioids.EncounterKey IS NOT NULL THEN 'Y' ELSE 'N' END AS Opioids,
	CASE WHEN ssri.EncounterKey IS NOT NULL THEN 'Y' ELSE 'N' END AS SSRI,
	CASE WHEN antipsych.EncounterKey IS NOT NULL THEN 'Y' ELSE 'N' END AS Antipsychotics,
	CASE WHEN lithium.EncounterKey IS NOT NULL THEN 'Y' ELSE 'N' END AS Lithium,
	CASE WHEN anticonv.EncounterKey IS NOT NULL THEN 'Y' ELSE 'N' END AS Anticonvulsants,
	CASE WHEN benzos.EncounterKey IS NOT NULL THEN 'Y' ELSE 'N' END AS Benzos,
	CASE WHEN nri.EncounterKey IS NOT NULL THEN 'Y' ELSE 'N' END AS NRI,
	CASE WHEN adhdstim.EncounterKey IS NOT NULL THEN 'Y' ELSE 'N' END AS ADHDStimulants,
	score5.Score AS Score5Score,
	score5.ScoreLevel AS Score5Level,
	score5.CalculationDateKey AS Score5Date,
	score49.Score AS Score49Score,
	score49.ScoreLevel AS Score49Level,
	score49.CalculationDateKey AS Score49Date,
	score50.Score AS Score50Score,
	score50.ScoreLevel AS Score50Level,
	score50.CalculationDateKey AS Score50Date,
	score51.Score AS Score51Score,
	score51.ScoreLevel AS Score51Level,
	score51.CalculationDateKey AS Score51Date,
	score52.Score AS Score52Score,
	score52.ScoreLevel AS Score52Level,
	score52.CalculationDateKey AS Score52Date,
	score53.Score AS Score53Score,
	score53.ScoreLevel AS Score53Level,
	score53.CalculationDateKey AS Score53Date,
	score54.Score AS Score54Score,
	score54.ScoreLevel AS Score54Level,
	score54.CalculationDateKey AS Score54Date,
	CASE WHEN depression.EncounterKey IS NOT NULL THEN 'Y' ELSE 'N' END AS Depression,
	CASE WHEN depressionplus.EncounterKey IS NOT NULL THEN 'Y' ELSE 'N' END AS DepressionPlus,
	CASE WHEN mddactive.EncounterKey IS NOT NULL THEN 'Y' ELSE 'N' END AS MDDActive,
	CASE WHEN mddremission.EncounterKey IS NOT NULL THEN 'Y' ELSE 'N' END AS MDDRemission,
	CASE WHEN anxietyplus.EncounterKey IS NOT NULL THEN 'Y' ELSE 'N' END AS AnxietyPlus,
	CASE WHEN anxiety.EncounterKey IS NOT NULL THEN 'Y' ELSE 'N' END AS Anxiety,
	CASE WHEN adjstress.EncounterKey IS NOT NULL THEN 'Y' ELSE 'N' END AS AdjStressDisorder,
	CASE WHEN mood.EncounterKey IS NOT NULL THEN 'Y' ELSE 'N' END AS MoodDisorders,
	CASE WHEN psychoticplus.EncounterKey IS NOT NULL THEN 'Y' ELSE 'N' END AS PsychoticPlus,
	CASE WHEN schizophrenia.EncounterKey IS NOT NULL THEN 'Y' ELSE 'N' END AS Schizophrenia,
	CASE WHEN personality.EncounterKey IS NOT NULL THEN 'Y' ELSE 'N' END AS PersDisorders,
	CASE WHEN mental.EncounterKey IS NOT NULL THEN 'Y' ELSE 'N' END AS MentalDisorder,
	CASE WHEN eating.EncounterKey IS NOT NULL THEN 'Y' ELSE 'N' END AS EatingDisorder,
	CASE WHEN sexual.EncounterKey IS NOT NULL THEN 'Y' ELSE 'N' END AS SexualDysfunction,
	CASE WHEN sleep.EncounterKey IS NOT NULL THEN 'Y' ELSE 'N' END AS SleepDisorder,
	CASE WHEN autism.EncounterKey IS NOT NULL THEN 'Y' ELSE 'N' END AS Autism,
	CASE WHEN ptsd.EncounterKey IS NOT NULL THEN 'Y' ELSE 'N' END AS PTSD,
	CASE WHEN ptsdplus.EncounterKey IS NOT NULL THEN 'Y' ELSE 'N' END AS PTSDPlus,
	CASE WHEN impulse.EncounterKey IS NOT NULL THEN 'Y' ELSE 'N' END AS ImpulseDisorders,
	CASE WHEN externalizing.EncounterKey IS NOT NULL THEN 'Y' ELSE 'N' END AS ExternalizingDis,
	CASE WHEN adhd.EncounterKey IS NOT NULL THEN 'Y' ELSE 'N' END AS ADHD,
	CASE WHEN malneoplasm.EncounterKey IS NOT NULL THEN 'Y' ELSE 'N' END AS MalignantNeoplasm,
	ISNULL(edcount.EDVisitCount, 0) AS EDVisitCount,
	eddxprim.EDPrimaryDx AS EDPrimDx,
	painintensity.NumericResponse AS PainIntensityScore,
	painintensity.ResponseDateKey AS PainIntensityDateKey,
	vas.NumericResponse AS VASScore,
	vas.ResponseDateKey AS VASDateKey,
	fatigue.NumericResponse AS FatigueScore,
	fatigue.ResponseDateKey AS FatigueDateKey,
	paininterf.NumericResponse AS PainInterfScore,
	paininterf.ResponseDateKey AS PainInterfDateKey,
	ISNULL(px.PxCount, 0) AS PxCount,
	CASE WHEN prescripmed.PresMedCount BETWEEN 4 AND 5 THEN 'Y' ELSE 'N' END AS PrescripMed45,	-- Between 4 and 5 prescribed medications in the last year before the encounter
	CASE WHEN prescripmed.PresMedCount > 5 THEN 'Y' ELSE 'N' END AS PrescripMedGT5,	-- More than 5 presciribed medications in the last year before the encounter
	pulmonary.MaxYear AS Pulmonary,
	fastgluc.MaxYear AS FastingGlucose,
	spotgluc.MaxYear AS SpotGlucose,
	lipid.MaxYear AS LipidLevels,
	crp.MaxYear AS CRP,
	cortisol.MaxYear AS Cortisol,
	thyroid.MaxYear AS Thyroid
INTO #BipolarDepression_EncLevelVars
FROM #TempBipolarDepressionEnc e
JOIN dbo.DurationDim age
	ON e.EncAgeKey = age.DurationKey
LEFT JOIN #TempBipolarDepressionEncWeight ew
	ON e.EncounterKey = ew.EncounterKey
LEFT JOIN #TempBipolarDepressionPastWeight pw
	ON e.EncounterKey = pw.EncounterKey
LEFT JOIN #TempBipolarDepressionEncHeight eh
	ON e.EncounterKey = eh.EncounterKey
LEFT JOIN #TempBipolarDepressionEncBMI eb
	ON e.EncounterKey = eb.EncounterKey
LEFT JOIN #TempBipolarDepressionHealthInsLoss hil
	ON e.EncounterKey = hil.EncounterKey
LEFT JOIN #TempBipolarDepressionPsychiatry psychiatry
	ON e.EncounterKey = psychiatry.EncounterKey
LEFT JOIN #TempBipolarDepressionPsychology psychology
	ON e.EncounterKey = psychology.EncounterKey
LEFT JOIN #TempBipolarDepressionVisitDxPrimary dxprimary
	ON e.EncounterKey = dxprimary.EncounterKey
LEFT JOIN #TempBipolarDepressionVisitDxNonPrimary dxnonprimary
	ON e.EncounterKey = dxnonprimary.EncounterKey
LEFT JOIN #TempBipolarDepressionHxSuicideAttempt hxsa
	ON e.PatientDurableKey = hxsa.PatientDurableKey
LEFT JOIN #TempSuicideAttemptEnc priorsaenc
	ON e.PatientDurableKey = priorsaenc.PatientDurableKey
	AND priorsaenc.EncDateKey < e.EncDateKey
LEFT JOIN #TempBipolarDepressionOpioids opioids
	ON e.EncounterKey = opioids.EncounterKey
LEFT JOIN #TempBipolarDepressionSSRI ssri
	ON e.EncounterKey = ssri.EncounterKey
LEFT JOIN #TempBipolarDepressionAntipsychotics antipsych
	ON e.EncounterKey = antipsych.EncounterKey
LEFT JOIN #TempBipolarDepressionLithium lithium
	ON e.EncounterKey = lithium.EncounterKey
LEFT JOIN #TempBipolarDepressionAnticonvulsants anticonv
	ON e.EncounterKey = anticonv.EncounterKey
LEFT JOIN #TempBipolarDepressionBenzos benzos
	ON e.EncounterKey = benzos.EncounterKey
LEFT JOIN #TempBipolarDepressionNRI nri
	ON e.EncounterKey = nri.EncounterKey
LEFT JOIN #TempBipolarDepressionADHD adhdstim
	ON e.EncounterKey = adhdstim.EncounterKey
LEFT JOIN #TempBipolarDepressionRiskScores score5
	ON e.EncounterKey = score5.EncounterKey
	AND score5.RiskScoreRuleKey = 5
LEFT JOIN #TempBipolarDepressionRiskScores score49
	ON e.EncounterKey = score49.EncounterKey
	AND score49.RiskScoreRuleKey = 49
LEFT JOIN #TempBipolarDepressionRiskScores score50
	ON e.EncounterKey = score50.EncounterKey
	AND score50.RiskScoreRuleKey = 50
LEFT JOIN #TempBipolarDepressionRiskScores score51
	ON e.EncounterKey = score51.EncounterKey
	AND score51.RiskScoreRuleKey = 51
LEFT JOIN #TempBipolarDepressionRiskScores score52
	ON e.EncounterKey = score52.EncounterKey
	AND score52.RiskScoreRuleKey = 52
LEFT JOIN #TempBipolarDepressionRiskScores score53
	ON e.EncounterKey = score53.EncounterKey
	AND score53.RiskScoreRuleKey = 53
LEFT JOIN #TempBipolarDepressionRiskScores score54
	ON e.EncounterKey = score54.EncounterKey
	AND score54.RiskScoreRuleKey = 54
LEFT JOIN #TempBipolarDepressionEncConditions  depression
        ON e.EncounterKey = depression.EncounterKey
        AND depression.Category = 'Depression'
LEFT JOIN #TempBipolarDepressionEncConditions  depressionplus
        ON e.EncounterKey = depressionplus.EncounterKey
        AND depressionplus.Category = 'Depression+'
LEFT JOIN #TempBipolarDepressionEncConditions  mddactive
        ON e.EncounterKey = mddactive.EncounterKey
        AND mddactive.Category = 'MDDActive'
LEFT JOIN #TempBipolarDepressionEncConditions  mddremission
        ON e.EncounterKey = mddremission.EncounterKey
        AND mddremission.Category = 'MDDRemission'
LEFT JOIN #TempBipolarDepressionEncConditions  anxietyplus
        ON e.EncounterKey = anxietyplus.EncounterKey
        AND anxietyplus.Category = 'Anxiety+'
LEFT JOIN #TempBipolarDepressionEncConditions  anxiety
        ON e.EncounterKey = anxiety.EncounterKey
        AND anxiety.Category = 'Anxiety'
LEFT JOIN #TempBipolarDepressionEncConditions  adjstress
        ON e.EncounterKey = adjstress.EncounterKey
        AND adjstress.Category = 'Adjustment/stress disorder'
LEFT JOIN #TempBipolarDepressionEncConditions  mood
        ON e.EncounterKey = mood.EncounterKey
        AND mood.Category = 'Mood Disorders, general'
LEFT JOIN #TempBipolarDepressionEncConditions  psychoticplus
        ON e.EncounterKey = psychoticplus.EncounterKey
        AND psychoticplus.Category = 'Psychotic+'
LEFT JOIN #TempBipolarDepressionEncConditions  schizophrenia
        ON e.EncounterKey = schizophrenia.EncounterKey
        AND schizophrenia.Category = 'Schizophrenia'
LEFT JOIN #TempBipolarDepressionEncConditions  personality
        ON e.EncounterKey = personality.EncounterKey
        AND personality.Category = 'Personality disorders'
LEFT JOIN #TempBipolarDepressionEncConditions  mental
        ON e.EncounterKey = mental.EncounterKey
        AND mental.Category = 'Mental disorder d.t. Substance Use'
LEFT JOIN #TempBipolarDepressionEncConditions  eating
        ON e.EncounterKey = eating.EncounterKey
        AND eating.Category = 'Eating disorder'
LEFT JOIN #TempBipolarDepressionEncConditions  sexual
        ON e.EncounterKey = sexual.EncounterKey
        AND sexual.Category = 'Sexual dysfunction'
LEFT JOIN #TempBipolarDepressionEncConditions  sleep
        ON e.EncounterKey = sleep.EncounterKey
        AND sleep.Category = 'Sleep disorder'
LEFT JOIN #TempBipolarDepressionEncConditions  autism
        ON e.EncounterKey = autism.EncounterKey
        AND autism.Category = 'Autism'
LEFT JOIN #TempBipolarDepressionEncConditions  ptsd
        ON e.EncounterKey = ptsd.EncounterKey
        AND ptsd.Category = 'PTSD'
LEFT JOIN #TempBipolarDepressionEncConditions  ptsdplus
        ON e.EncounterKey = ptsdplus.EncounterKey
        AND ptsdplus.Category = 'PTSD+'
LEFT JOIN #TempBipolarDepressionEncConditions  impulse
        ON e.EncounterKey = impulse.EncounterKey
        AND impulse.Category = 'Impulse disorders'
LEFT JOIN #TempBipolarDepressionEncConditions  externalizing
        ON e.EncounterKey = externalizing.EncounterKey
        AND externalizing.Category = 'Externalizing Disorders'
LEFT JOIN #TempBipolarDepressionEncConditions  adhd
        ON e.EncounterKey = adhd.EncounterKey
        AND adhd.Category = 'ADHD'
LEFT JOIN #TempBipolarDepressionEncConditions  malneoplasm
        ON e.EncounterKey = malneoplasm.EncounterKey
        AND malneoplasm.Category = 'Malignant neoplasm'
LEFT JOIN #TempBipolarDepressionEDVisits edcount
	ON e.EncounterKey = edcount.EncounterKey
LEFT JOIN #TempBipolarDepressionEDDxPrimary eddxprim
	ON e.EncounterKey = eddxprim.EncounterKey
LEFT JOIN #TempBipolarDepressionPRO  painintensity
        ON e.EncounterKey = painintensity.EncounterKey
        AND painintensity.Measure = 'PROMIS Pain Intensity 3A'
LEFT JOIN #TempBipolarDepressionPRO  vas
        ON e.EncounterKey = vas.EncounterKey
        AND vas.Measure = 'VAS'
LEFT JOIN #TempBipolarDepressionPRO  fatigue
        ON e.EncounterKey = fatigue.EncounterKey
        AND fatigue.Measure = 'PROMIS Fatigue'
LEFT JOIN #TempBipolarDepressionPRO  paininterf
        ON e.EncounterKey = paininterf.EncounterKey
        AND paininterf.Measure = 'PROMIS Pain Interference'
LEFT JOIN #TempBipolarDepressionPxCount px
	ON e.EncounterKey = px.EncounterKey
LEFT JOIN #TempBipolarDepressionPrescripMedCount prescripmed
	ON e.EncounterKey = prescripmed.EncounterKey
LEFT JOIN #TempBipolarDepressionLabs  pulmonary
        ON e.EncounterKey = pulmonary.EncounterKey
        AND pulmonary.LabType = 'Pulmonary function test'
LEFT JOIN #TempBipolarDepressionLabs  fastgluc
        ON e.EncounterKey = fastgluc.EncounterKey
        AND fastgluc.LabType = 'Fasting glucose'
LEFT JOIN #TempBipolarDepressionLabs  spotgluc
        ON e.EncounterKey = spotgluc.EncounterKey
        AND spotgluc.LabType = 'Spot glucose'
LEFT JOIN #TempBipolarDepressionLabs  lipid
        ON e.EncounterKey = lipid.EncounterKey
        AND lipid.LabType = 'Lipid levels'
LEFT JOIN #TempBipolarDepressionLabs  crp
        ON e.EncounterKey = crp.EncounterKey
        AND crp.LabType = 'CRP'
LEFT JOIN #TempBipolarDepressionLabs  cortisol
        ON e.EncounterKey = cortisol.EncounterKey
        AND cortisol.LabType = 'Cortisol'
LEFT JOIN #TempBipolarDepressionLabs  thyroid
        ON e.EncounterKey = thyroid.EncounterKey
        AND thyroid.LabType = 'Thyroid'
GO
DECLARE @timestamp nvarchar(50)
SET @timestamp = convert(nvarchar(50), GETDATE(), 121)
RAISERROR('%s: Finished encounter-level variables', 10, 1, @timestamp) WITH NOWAIT
GO

drop table if exists #TransferredEnc;

select a.EncounterKey
into #TransferredEnc
from #BipolarDepression_EncLevelVars a
where 1=2
;

drop table if exists #BatchInsertLog;
-- Temp table to log batch times
CREATE TABLE #BatchInsertLog (
    batch_number INT,
    insert_time DATETIME
);

-- Declare control variables
DECLARE @BatchSize INT = 10000;
DECLARE @BatchCount INT = 0;
DECLARE @InsertTime DATETIME;
DECLARE @msg NVARCHAR(100);

WHILE 1 = 1
BEGIN
    -- Clean up any previous staging data
    DROP TABLE if exists #BatchData;

    -- Materialize current batch into a temp table
    SELECT TOP (@BatchSize) *
    INTO #BatchData
    FROM #BipolarDepression_EncLevelVars s
    WHERE NOT EXISTS (
        SELECT 1 FROM #TransferredEnc t WHERE t.EncounterKey = s.EncounterKey
    )
    ORDER BY s.EncounterKey;

    -- Exit loop if there's nothing to insert
    IF NOT EXISTS (SELECT 1 FROM #BatchData)
        BREAK;

    -- Capture timestamp before insert
    SET @InsertTime = GETDATE();

    -- Insert into target table
    INSERT INTO PROJECTS.ProjectD52021F.[dbo].[BipolarDepression_EncLevelVars]
    SELECT *
    FROM #BatchData;

    -- Track inserted patient keys
    INSERT INTO #TransferredEnc (EncounterKey)
    SELECT EncounterKey FROM #BatchData;

    -- Increment batch counter
    SET @BatchCount += 1;

    -- Log the batch (optional)
    INSERT INTO #BatchInsertLog (batch_number, insert_time)
    VALUES (@BatchCount, @InsertTime);

    -- Print progress
	SET @msg = 'Batch ' + CAST(@BatchCount AS NVARCHAR) + ' inserted at ' + CONVERT(NVARCHAR(30), @InsertTime, 120);
	RAISERROR('%s', 0, 1, @msg) WITH NOWAIT;

END

-- view the batch log
SELECT * FROM #BatchInsertLog;

SELECT COUNT_BIG(*) FROM #BipolarDepression_EncLevelVars

/* Use this workaround if timeout occurs
Run on the COSMOS server in a separate tab; if it still does not work, close the tab, re-open a new one and repeat
SELECT TOP 10 * FROM PROJECTS.ProjectD0F7BC.dbo.MV_DepressionPlus_Test
*/

