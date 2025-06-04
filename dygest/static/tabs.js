// Back to Top Button
window.onscroll = function() { scrollFunction() };

function scrollFunction() {
    var backToTopButton = document.getElementById("back-to-top");
    if (document.body.scrollTop > 500 || document.documentElement.scrollTop > 500) {
        backToTopButton.style.display = "block";
    } else {
        backToTopButton.style.display = "none";
    }
}

function scrollToTop() {
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

document.addEventListener('DOMContentLoaded', function() {
    const pageContainer = document.querySelector('.page-container');
    if (!pageContainer) {
        console.error('Page container not found.');
        return;
    }

    const tabbedContentConfigs = [
        { class: 'summary', title: 'Summary', idPrefix: 'summary' },
        { class: 'toc', title: 'Table of Contents', idPrefix: 'toc' },
        { class: 'keywords', title: 'Keywords', idPrefix: 'keywords' },
        { class: 'metadata', title: 'Metadata', idPrefix: 'metadata' }
    ];

    const tabBar = document.createElement('div');
    tabBar.className = 'dgst-tab-bar';

    const contentPanels = [];

    let firstTabSet = false;
    tabbedContentConfigs.forEach((config, index) => {
        const contentDiv = pageContainer.querySelector('.' + config.class);
        if (contentDiv) {
            const panelId = `dgst-${config.idPrefix}-panel`;
            contentDiv.id = panelId;
            contentDiv.classList.add('dgst-tab-content-panel');
            contentPanels.push(contentDiv);

            const tabLink = document.createElement('button');
            tabLink.className = 'dgst-tab-link';
            tabLink.textContent = config.title;
            tabLink.setAttribute('data-tab-target', '#' + panelId);
            tabLink.setAttribute('role', 'tab');
            tabLink.setAttribute('aria-controls', panelId);
            tabLink.setAttribute('aria-selected', 'false');

            tabLink.addEventListener('click', () => {
                // Deactivate all tabs and panels
                document.querySelectorAll('.dgst-tab-link').forEach(link => {
                    link.classList.remove('active');
                    link.setAttribute('aria-selected', 'false');
                });
                document.querySelectorAll('.dgst-tab-content-panel').forEach(panel => {
                    panel.classList.remove('active');
                });

                // Activate clicked tab and corresponding panel
                tabLink.classList.add('active');
                tabLink.setAttribute('aria-selected', 'true');
                const targetPanel = document.querySelector(tabLink.getAttribute('data-tab-target'));
                if (targetPanel) {
                    targetPanel.classList.add('active');
                }
            });

            tabBar.appendChild(tabLink);

            // Make the first *found* tab active by default
            if (!firstTabSet) {
                tabLink.classList.add('active');
                tabLink.setAttribute('aria-selected', 'true');
                contentDiv.classList.add('active');
                firstTabSet = true;
            } else {
                contentDiv.classList.remove('active');
            }
        } else {
            console.warn(`Content div with class "${config.class}" not found.`);
        }
    });

    // Insert the tab bar before the first tabbed content div (or at the top of pageContainer if no specific order)
    const mainContentDiv = pageContainer.querySelector('.page-container > .content');

    if (contentPanels.length > 0) {
        const firstPanel = contentPanels[0];
        const insertionParent = firstPanel.parentNode;
        insertionParent.insertBefore(tabBar, firstPanel);
    } else if (mainContentDiv) {
        pageContainer.insertBefore(tabBar, mainContentDiv);
    } else {
        pageContainer.appendChild(tabBar);
    }

    // Ensure tab bar is shown even if only one tab exists
    if (contentPanels.length === 1) {
        // Force display in case a stylesheet hides single-tab bars
        tabBar.style.display = tabBar.style.display || 'flex';
    }

    // Back to Top button visibility
    const backToTopButton = document.getElementById('back-to-top');
    if (backToTopButton) {
        if (typeof scrollToTop !== 'function') {
            window.scrollToTop = function() {
                window.scrollTo({ top: 0, behavior: 'smooth' });
            };
        }

        window.addEventListener('scroll', () => {
            if (window.pageYOffset > 300) { // Show button after scrolling 300px
                backToTopButton.style.display = 'block';
            } else {
                backToTopButton.style.display = 'none';
            }
        });
    }
});