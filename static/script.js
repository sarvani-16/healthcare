document.addEventListener("DOMContentLoaded", function () {

    console.log("VitalSign frontend loaded successfully.");

    // Add active navigation class
    const currentPath = window.location.pathname;

    const navLinks = document.querySelectorAll(".nav-link");

    navLinks.forEach(function (link) {

        const href = link.getAttribute("href");

        if (
            href === currentPath ||
            (currentPath === "/" && href === "/")
        ) {

            link.classList.add("active");

        }

    });

});