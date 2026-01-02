// ICEBURG Landing Page Logic

document.addEventListener('DOMContentLoaded', () => {
    // 1. Telemetry Simulation
    const telemetryElements = document.querySelectorAll('.stat-value');

    function updateTelemetry() {
        telemetryElements.forEach(el => {
            if (el.dataset.type === 'cpu') {
                el.innerText = (Math.random() * (45 - 15) + 15).toFixed(1) + '%';
            } else if (el.dataset.type === 'mem') {
                el.innerText = (Math.random() * (12 - 4) + 4).toFixed(1) + 'GB';
            } else if (el.dataset.type === 'net') {
                el.innerText = (Math.random() * (100 - 20) + 20).toFixed(0) + ' MS';
            }
        });
    }

    setInterval(updateTelemetry, 2000);

    // 2. Parallax Effect on Grid
    document.addEventListener('mousemove', (e) => {
        const grid = document.querySelector('.grid-lines');
        if (grid) {
            const x = e.clientX / window.innerWidth;
            const y = e.clientY / window.innerHeight;

            grid.style.transform = `translate(${x * 20}px, ${y * 20}px)`;
        }
    });

    // 3. Entry Transition
    const enterBtn = document.querySelector('.enter-button');
    enterBtn.addEventListener('click', (e) => {
        e.preventDefault();
        const target = enterBtn.getAttribute('href');

        document.body.style.opacity = '0';
        document.body.style.transition = 'opacity 0.8s cubic-bezier(0.16, 1, 0.3, 1)';

        setTimeout(() => {
            window.location.href = target;
        }, 800);
    });

    // 4. Three.js Background (Placeholder for future expansion)
    // Currently using CSS grid for performance solidity
});
