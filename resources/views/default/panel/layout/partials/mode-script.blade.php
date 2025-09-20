<script>
    (() => {
        const lqdDarkMode = localStorage.getItem("lqdDarkMode");
        const navbarIsShrinked = localStorage.getItem("lqdNavbarShrinked");

        document.body.classList.toggle("theme-dark", lqdDarkMode == "true");
        document.body.classList.toggle("theme-light", lqdDarkMode != "true");

        if (navbarIsShrinked === "true") {
            document.body.classList.add("navbar-shrinked");
        }
    })();
</script>

@auth
    <script>
        (() => {
            const currentTheme = document.querySelector('body').getAttribute('data-theme');
            const focusModeEnabled = localStorage.getItem(currentTheme + ":lqdFocusModeEnabled");

            document.body.classList.toggle("focus-mode", focusModeEnabled == "true");

            // TODO: output this if customizer is enabled
            const customizerEdits = localStorage.getItem(`${currentTheme}:lqdCustomizerStyle`);
            if (customizerEdits) {
                const styleTag = document.querySelector('#lqd-customizer-style');

                if (styleTag) {
                    styleTag.innerText = customizerEdits;
                }
            }
        })
        ();
    </script>
@endauth
