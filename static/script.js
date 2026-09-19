/**
 * VitalSign - Healthcare Readmission Prediction System
 * Interactive Client Script
 */

document.addEventListener("DOMContentLoaded", function () {
    console.log("VitalSign Clinical ML Platform initialized.");

    // Highlight active sidebar navigation link
    const currentPath = window.location.pathname;
    const sideLinks = document.querySelectorAll(".side-link");

    sideLinks.forEach(function (link) {
        const href = link.getAttribute("href");
        if (
            href === currentPath ||
            (currentPath === "/" && (href === "/" || href === "/index"))
        ) {
            link.style.background = "var(--sidebar-light)";
            link.style.color = "#ffffff";
            link.style.fontWeight = "700";
            link.style.borderLeft = "3px solid #2563eb";
        }
    });

    // Auto-dismiss flash messages after 5 seconds
    const flashMessages = document.querySelectorAll(".flash");
    flashMessages.forEach(function (flash) {
        flash.style.cursor = "pointer";
        flash.title = "Click to dismiss";
        flash.addEventListener("click", function () {
            flash.style.opacity = "0";
            setTimeout(() => flash.remove(), 300);
        });

        setTimeout(function () {
            flash.style.transition = "opacity 0.5s ease";
            flash.style.opacity = "0";
            setTimeout(() => flash.remove(), 500);
        }, 6000);
    });
});

// Sample Patient Autofill for Prediction Demonstration
function loadSampleLowRisk() {
    const setVal = (id, val) => {
        const el = document.getElementById(id);
        if (el) el.value = val;
    };
    setVal('age', '[30-40)');
    setVal('gender', 'Female');
    setVal('race', 'Caucasian');
    setVal('admission_type_id', '3'); // Elective
    setVal('discharge_disposition_id', '1'); // Home
    setVal('admission_source_id', '1'); // Physician Referral
    setVal('time_in_hospital', '2');
    setVal('num_lab_procedures', '24');
    setVal('num_procedures', '1');
    setVal('num_medications', '8');
    setVal('number_outpatient', '0');
    setVal('number_emergency', '0');
    setVal('number_inpatient', '0');
    setVal('number_diagnoses', '4');
    setVal('insulin', 'No');
    setVal('change', 'No');
    setVal('diabetesMed', 'Yes');
}

function loadSampleHighRisk() {
    const setVal = (id, val) => {
        const el = document.getElementById(id);
        if (el) el.value = val;
    };
    setVal('age', '[70-80)');
    setVal('gender', 'Male');
    setVal('race', 'AfricanAmerican');
    setVal('admission_type_id', '1'); // Emergency
    setVal('discharge_disposition_id', '3'); // SNF
    setVal('admission_source_id', '7'); // ER
    setVal('time_in_hospital', '10');
    setVal('num_lab_procedures', '75');
    setVal('num_procedures', '4');
    setVal('num_medications', '28');
    setVal('number_outpatient', '2');
    setVal('number_emergency', '3');
    setVal('number_inpatient', '4');
    setVal('number_diagnoses', '9');
    setVal('insulin', 'Up');
    setVal('change', 'Ch');
    setVal('diabetesMed', 'Yes');
}