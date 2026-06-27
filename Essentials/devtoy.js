(function () {
    "use strict";

    const PASTIZZI_SRC = "/devtoy-assets/pastizzi.png";
    const FIREWORK_TRAIL_SRC = "/devtoy-assets/firework-trail.png";
    const FIREWORK_BURST_SRC = "/devtoy-assets/firework-burst.png";

    function randomColor() {
        return `hsl(${Math.floor(Math.random() * 360)} ${55 + Math.random() * 35}% ${35 + Math.random() * 45}%)`;
    }

    function randomFireworkFilter() {
        const hue = Math.floor(Math.random() * 360);
        const saturation = 1.15 + Math.random() * 0.75;
        const brightness = 0.92 + Math.random() * 0.28;
        return `hue-rotate(${hue}deg) saturate(${saturation}) brightness(${brightness}) drop-shadow(0 0 14px hsl(${hue} 100% 70% / 0.85))`;
    }

    function createDevToy({ elements, t, flags }) {
        let pastizziAnimation = null;
        let partyMode = false;
        let gravityMode = false;
        let gravityAnimation = null;
        let draggedPastizz = null;
        let draggedGravityItem = null;
        const pastizzi = [];
        const gravityItems = [];

        function rememberPartyStyle(element) {
            if (element.dataset.partyStored === "1") {
                return;
            }

            element.dataset.partyStored = "1";
            element.dataset.partyColor = element.style.color || "";
            element.dataset.partyBackgroundColor = element.style.backgroundColor || "";
            element.dataset.partyBorderColor = element.style.borderColor || "";
            element.dataset.partyBoxShadow = element.style.boxShadow || "";
        }

        function restorePartyStyle(element) {
            if (element.dataset.partyStored !== "1") {
                return;
            }

            element.style.color = element.dataset.partyColor || "";
            element.style.backgroundColor = element.dataset.partyBackgroundColor || "";
            element.style.borderColor = element.dataset.partyBorderColor || "";
            element.style.boxShadow = element.dataset.partyBoxShadow || "";
            delete element.dataset.partyStored;
            delete element.dataset.partyColor;
            delete element.dataset.partyBackgroundColor;
            delete element.dataset.partyBorderColor;
            delete element.dataset.partyBoxShadow;
        }

        function paintPartyGroup(selector, painter) {
            document.querySelectorAll(selector).forEach((element) => {
                rememberPartyStyle(element);
                painter(element);
            });
        }

        function setPartyMode(enabled) {
            partyMode = enabled;

            if (!enabled) {
                document.querySelectorAll("[data-party-stored='1']").forEach(restorePartyStyle);
                updateModeButtons();
                return;
            }

            const pageText = randomColor();
            rememberPartyStyle(document.body);
            document.body.style.color = pageText;
            document.body.style.backgroundColor = randomColor();

            paintPartyGroup(".panel", (element) => {
                element.style.backgroundColor = randomColor();
                element.style.borderColor = randomColor();
                element.style.color = randomColor();
                element.style.boxShadow = `0 8px 24px ${randomColor()}`;
            });

            paintPartyGroup("textarea, input, .result-box, pre, .suggestion-popover", (element) => {
                element.style.backgroundColor = randomColor();
                element.style.borderColor = randomColor();
                element.style.color = randomColor();
            });

            paintPartyGroup("button", (element) => {
                element.style.backgroundColor = randomColor();
                element.style.borderColor = randomColor();
                element.style.color = randomColor();
            });

            paintPartyGroup("h1, p, label, .status, .correction-timer", (element) => {
                element.style.color = randomColor();
            });

            paintPartyGroup(".ambiguous-word, .english-word", (element) => {
                element.style.backgroundColor = randomColor();
                element.style.color = randomColor();
            });

            updateModeButtons();
        }

        function togglePartyMode() {
            setPartyMode(!partyMode);
        }

        function updateModeButtons() {
            elements.pastizziModeButton.style.display =
                flags.enablePastizziMode && flags.showPastizziDevtoolsButton ? "" : "none";
            elements.fireworksButton.style.display = flags.showFireworksDevtoolsButton ? "" : "none";
            elements.partyModeButton.style.display = flags.showPartyDevtoolsButton ? "" : "none";
            elements.gravityModeButton.style.display = flags.showGravityDevtoolsButton ? "" : "none";

            elements.pastizziModeButton.textContent = t("pastizziModeOff");
            elements.fireworksButton.textContent = t("fireworksButton");
            elements.partyModeButton.textContent = partyMode
                ? t("partyModeOn")
                : t("partyModeOff");
            elements.gravityModeButton.textContent = gravityMode
                ? t("gravityModeOn")
                : t("gravityModeOff");

            elements.pastizziModeButton.classList.remove("active");
            elements.fireworksButton.classList.remove("active");
            elements.partyModeButton.classList.toggle("active", partyMode);
            elements.gravityModeButton.classList.toggle("active", gravityMode);
        }

        function startPastizziAnimation() {
            if (pastizziAnimation !== null) {
                return;
            }

            let lastTime = performance.now();

            function tick(now) {
                const dt = Math.min(32, now - lastTime) / 1000;
                lastTime = now;
                const floor = window.innerHeight - 10;

                pastizzi.forEach((item) => {
                    if (!item.dragging) {
                        item.vy += 1300 * dt;
                        item.vx *= 0.992;
                        item.vy *= 0.996;
                        item.angularVelocity *= 0.993;
                        item.x += item.vx * dt;
                        item.y += item.vy * dt;
                        item.rotation += item.angularVelocity * dt;

                        const half = item.size / 2;
                        if (item.x < half) {
                            item.x = half;
                            item.vx = Math.abs(item.vx) * 0.62;
                        }
                        if (item.x > window.innerWidth - half) {
                            item.x = window.innerWidth - half;
                            item.vx = -Math.abs(item.vx) * 0.62;
                        }
                        if (item.y > floor - half) {
                            item.y = floor - half;
                            item.vy = -Math.abs(item.vy) * 0.24;
                            item.vx *= 0.86;
                            if (Math.abs(item.vy) < 45) {
                                item.vy = 0;
                            }
                        }
                    }

                    item.element.style.transform = `translate(${item.x - item.size / 2}px, ${item.y - item.size / 2}px) rotate(${item.rotation}deg)`;
                });

                pastizziAnimation = window.requestAnimationFrame(tick);
            }

            pastizziAnimation = window.requestAnimationFrame(tick);
        }

        function makePastizzInteractive(item) {
            const element = item.element;

            element.addEventListener("pointerdown", (event) => {
                event.preventDefault();
                draggedPastizz = item;
                item.dragging = true;
                item.dragOffsetX = event.clientX - item.x;
                item.dragOffsetY = event.clientY - item.y;
                item.lastPointerX = event.clientX;
                item.lastPointerY = event.clientY;
                item.lastPointerAt = performance.now();
                item.vx = 0;
                item.vy = 0;
                element.style.cursor = "grabbing";
                element.setPointerCapture(event.pointerId);
            });

            element.addEventListener("pointermove", (event) => {
                if (draggedPastizz !== item) {
                    return;
                }

                const now = performance.now();
                const dt = Math.max(1, now - item.lastPointerAt) / 1000;
                const nextX = event.clientX - item.dragOffsetX;
                const nextY = event.clientY - item.dragOffsetY;

                item.vx = (event.clientX - item.lastPointerX) / dt;
                item.vy = (event.clientY - item.lastPointerY) / dt;
                item.angularVelocity = item.vx * 0.35;
                item.x = nextX;
                item.y = nextY;
                item.lastPointerX = event.clientX;
                item.lastPointerY = event.clientY;
                item.lastPointerAt = now;
            });

            element.addEventListener("pointerup", () => {
                item.dragging = false;
                draggedPastizz = null;
                element.style.cursor = "grab";
            });

            element.addEventListener("pointercancel", () => {
                item.dragging = false;
                draggedPastizz = null;
                element.style.cursor = "grab";
            });
        }

        function spawnPastizz() {
            if (!flags.enablePastizziMode) {
                return;
            }

            const size = Math.random() * 60 + 28;
            const element = document.createElement("img");
            element.className = "pastizzi devtoy-pastizzi";
            element.src = PASTIZZI_SRC;
            element.draggable = false;
            element.style.width = `${size}px`;
            element.style.position = "fixed";
            element.style.left = "0";
            element.style.top = "0";
            element.style.zIndex = "9999";
            element.style.animation = "none";
            element.style.pointerEvents = "auto";
            element.style.cursor = "grab";
            element.style.touchAction = "none";
            element.style.userSelect = "none";
            element.style.willChange = "transform";

            const item = {
                element,
                size,
                x: Math.random() * window.innerWidth,
                y: -size,
                vx: (Math.random() - 0.5) * 260,
                vy: Math.random() * 120,
                rotation: Math.random() * 360,
                angularVelocity: (Math.random() - 0.5) * 720,
                dragging: false,
            };

            pastizzi.push(item);
            document.body.appendChild(element);
            makePastizzInteractive(item);
            startPastizziAnimation();

            while (pastizzi.length > 140) {
                const old = pastizzi.shift();
                old.element.remove();
            }
        }

        function launchPastizzi(count = 50) { // Pastizzi per click; raise for denser rain, lower for lighter waves.
            if (!flags.enablePastizziMode) {
                return;
            }

            for (let i = 0; i < count; i++) {
                window.setTimeout(spawnPastizz, Math.random() * 2000); // 2000ms spreads one wave over 2 seconds.
            }
        }

        function spawnFirework() {
            const tint = randomFireworkFilter();

            // Firework path: widen the X drift for variety, but keep the image rotated to this exact vector.
            const startX = 32 + Math.random() * Math.max(80, window.innerWidth - 64); // 32px side padding keeps rockets off the screen edge.
            const startY = window.innerHeight + 80; // Starts below the viewport; increase for a later entrance from off-screen.
            const endX = Math.min(
                window.innerWidth - 60, // 60px right safety margin so bursts do not clip too hard.
                Math.max(60, startX + (Math.random() - 0.5) * 220) // 220px is sideways drift range; higher = more diagonal rockets.
            );
            const endY = 70 + Math.random() * Math.max(120, window.innerHeight * 0.42); // 70px top margin; 0.42 keeps bursts in upper half-ish.

            // Visual scale/timing knobs. Higher trailWidth/burstSize = bigger fireworks.
            const trailWidth = 42 + Math.random() * 26; // Rocket image width: 42-68px; image height scales with it.
            const burstSize = 230 + Math.random() * 170; // Burst image width: 230-400px; raise both numbers for larger explosions.
            const travelMs = 900 + Math.random() * 380; // Rocket travel duration: 780-1160ms; higher = slower ascent.
            const burstMs = 760 + Math.random() * 280; // Burst duration: 760-1040ms; higher = longer fade-out.
            const slowdownAt = 0.78; // Main ascent ends at 78%; remaining segment is the final slowdown/compression.
            const finalTrailFadeMs = 20; // Short final trail fade; preserves the previous fade timing.
            const finalTrailFadeAt = 1 - finalTrailFadeMs / travelMs; // Trail stays visible until this late offset, then fades before burst.
            const slowdownStartX = startX + (endX - startX) * 0.9; // Trail-head X where the slowdown/crossfade begins.
            const slowdownStartY = startY + (endY - startY) * 0.9; // Trail-head Y where the slowdown/crossfade begins.
            const pathAngle = Math.atan2(endX - startX, startY - endY) * 180 / Math.PI; // Converts travel vector to degrees so the trail points along motion.
            const wobble = (Math.random() - 0.5) * 7; // Small random tilt in degrees; lower for stricter alignment, higher for messier rockets.
            const rotation = pathAngle + wobble; // Final orientation applied to the trail image.

            // The source trail image has the bright firework head near the top; this keeps that head on the path.
            const trailHeadX = "-50%"; // Horizontal anchor for the image head; -50% centers the head on the path.
            const trailHeadY = "-18%"; // Vertical anchor for the bright head; adjust if a replacement trail image has a different head position.

            function trailTransform(x, y, scaleX, scaleY) {
                return `translate(${x}px, ${y}px) rotate(${rotation}deg) translate(${trailHeadX}, ${trailHeadY}) scale(${scaleX}, ${scaleY})`; // Order matters: move path point, rotate image, align head, then squash/stretch.
            }

            const trail = document.createElement("img");
            trail.className = "devtoy-firework-trail";
            trail.src = FIREWORK_TRAIL_SRC;
            trail.draggable = false;
            trail.style.position = "fixed";
            trail.style.left = "0";
            trail.style.top = "0";
            trail.style.width = `${trailWidth}px`; // Uses width only so the image keeps its natural proportions before animation scaling.
            trail.style.zIndex = "10000"; // Above the app UI; burst uses 10001 so it appears above the trail.
            trail.style.pointerEvents = "none";
            trail.style.userSelect = "none";
            trail.style.filter = tint;
            trail.style.willChange = "transform, opacity, filter"; // Hint for smoother animation; only use for actively animated pieces.
            trail.style.transformOrigin = "50% 18%"; // Rotation pivot near the firework head; match this to trailHeadX/Y.
            document.body.appendChild(trail);

            const trailAnimation = trail.animate(
                [
                    {
                        transform: trailTransform(startX, startY, 0.28, 0.42), // Tiny and vertically short at launch.
                        opacity: 0, // Fully invisible before the quick fade-in.
                    },
                    {
                        offset: 0.08, // Fade-in completes at 8% of the flight; raise for a slower visible fade.
                        transform: trailTransform(
                            startX + (endX - startX) * 0.08, // 8% along the X path when fade-in finishes.
                            startY + (endY - startY) * 0.08, // 8% along the Y path when fade-in finishes.
                            0.42, // Early width scale; higher = rocket grows sooner.
                            0.66 // Early height scale; higher = longer trail sooner.
                        ),
                        opacity: 1,
                    },
                    {
                        offset: slowdownAt, // Main ascent ends here; remaining segment is slowdown/compression.
                        transform: trailTransform(
                            slowdownStartX, // 90% of the path reached before the final slow curl.
                            slowdownStartY, // 90% of the path reached before the final slow curl.
                            1.05, // Near-full width just before the end.
                            1.05 // Near-full height just before the end; lower if trail feels too long.
                        ),
                        opacity: 1,
                    },
                    {
                        offset: finalTrailFadeAt, // Trail has reached the final endpoint and holds briefly before fading.
                        transform: trailTransform(endX, endY, 0.75, 0.05), // Endpoint shrink before fade; scaleY 0.5 keeps some body before it becomes the pop.
                        filter: `${tint} saturate(2.2) brightness(1.35)`,
                        opacity: 1,
                    },
                    {
                        transform: trailTransform(endX, endY, 0.75, 0.05), // Same endpoint shape as above; only opacity changes during the pop crossfade.
                        filter: `${tint} saturate(2.2) brightness(1.35)`,
                        opacity: 0.5, // Fade the trail out before the burst.
                    },
                ],
                {
                    duration: travelMs, // Total time for all trail keyframes.
                    easing: "cubic-bezier(.16,.86,.2,1)", // Fast launch with gentle slowdown; tweak curve for flight feel.
                    fill: "forwards",
                }
            );

            // Spawn the burst slightly before the trail ends so there is no dead gap.
            const burstOverlapMs = 180; // How many ms before trail end the burst appears.
            window.setTimeout(() => {
                trail.remove();

                const burst = document.createElement("img");
                burst.className = "devtoy-firework-burst";
                burst.src = FIREWORK_BURST_SRC;
                burst.draggable = false;
                burst.style.position = "fixed";
                burst.style.left = "0";
                burst.style.top = "0";
                burst.style.width = `${burstSize}px`; // Burst size chosen earlier; all burst scale values multiply this.
                burst.style.zIndex = "10001"; // One layer above the trail so it covers the final rocket head.
                burst.style.pointerEvents = "none";
                burst.style.userSelect = "none";
                burst.style.filter = tint;
                burst.style.willChange = "transform, opacity";
                document.body.appendChild(burst);

                // Burst animation: first keyframe pops it into place, middle holds brightness, final fades it out.
                const burstAnimation = burst.animate(
                    [
                        {
                            transform: `translate(${endX}px, ${endY}px) translate(-50%, -50%) scale(0.05)`, // Start tiny at rocket endpoint.
                            opacity: 1, // Starts almost fully visible for a sharp pop.
                        },
                        {
                            offset: 0.36, // Holds brightness until 36% of burst duration; lower = faster fade begins.
                            opacity: 1, // Peak brightness.
                        },
                        {
                            transform: `translate(${endX}px, ${endY}px) translate(-50%, -50%) scale(1.32)`, // Final expansion; higher = larger fading halo.
                            opacity: 0, // Fade away completely before removing the element.
                        },
                    ],
                    {
                        duration: burstMs, // Total burst expand/fade time.
                        easing: "cubic-bezier(.1,.72,.18,1)", // Quick expansion, soft ending.
                        fill: "forwards",
                    }
                );

                burstAnimation.onfinish = () => burst.remove();
            }, Math.max(0, travelMs - burstOverlapMs));
        }

        function launchFireworks(count = 6) { // Default fireworks per button press if caller does not override count.
            if (!flags.showFireworksDevtoolsButton) {
                return;
            }

            // Button batch size/timing: count controls how many fireworks, delay spacing controls the cascade.
            for (let i = 0; i < count; i++) {
                window.setTimeout(spawnFirework, i * 170 + Math.random() * 170); // 170ms base spacing + random jitter; lower = more simultaneous.
            }
        }

        function gravityTargets() {
            // Individual leaf-level selectors to capture as separate falling objects.
            // Order matters: more specific selectors listed first so parents are not
            // also captured as separate items when their children already are.
            const SELECTORS = [
                "button",
                "input",
                "textarea",
                "h1", "h2", "h3",
                "label",
                ".subtitle",
                ".status",
                ".correction-timer",
                "pre",
                ".result-box",
                ".devtoy-pastizzi",
            ];

            const already = new Set(gravityItems.map((item) => item.element));
            const collected = [];
            const seen = new Set();

            for (const sel of SELECTORS) {
                document.querySelectorAll(sel).forEach((el) => {
                    if (seen.has(el) || already.has(el)) return;
                    // Skip elements that are descendants of a button (the button itself will fall)
                    if (el.closest("button") && el.tagName.toLowerCase() !== "button") return;
                    const rect = el.getBoundingClientRect();
                    if (rect.width === 0 || rect.height === 0) return;
                    seen.add(el);
                    collected.push(el);
                });
            }

            return collected;
        }

        function makeGravityItemInteractive(item) {
            const element = item.element;

            element.style.cursor = "grab";
            element.style.touchAction = "none";
            element.style.userSelect = "none";

            element.addEventListener("pointerdown", (event) => {
                // Only drag with primary button or touch
                if (event.button !== 0 && event.pointerType === "mouse") return;
                event.preventDefault();
                draggedGravityItem = item;
                item.dragging = true;
                item.dragOffsetX = event.clientX - item.x;
                item.dragOffsetY = event.clientY - item.y;
                item.lastPointerX = event.clientX;
                item.lastPointerY = event.clientY;
                item.lastPointerAt = performance.now();
                item.vx = 0;
                item.vy = 0;
                element.style.cursor = "grabbing";
                element.setPointerCapture(event.pointerId);
            });

            element.addEventListener("pointermove", (event) => {
                if (draggedGravityItem !== item) return;

                const now = performance.now();
                const dt = Math.max(1, now - item.lastPointerAt) / 1000;
                item.vx = (event.clientX - item.lastPointerX) / dt;
                item.vy = (event.clientY - item.lastPointerY) / dt;
                item.angularVelocity = item.vx * 0.18;
                item.x = event.clientX - item.dragOffsetX;
                item.y = event.clientY - item.dragOffsetY;
                item.lastPointerX = event.clientX;
                item.lastPointerY = event.clientY;
                item.lastPointerAt = now;
            });

            function releaseGravity() {
                item.dragging = false;
                draggedGravityItem = null;
                element.style.cursor = "grab";
            }

            element.addEventListener("pointerup", releaseGravity);
            element.addEventListener("pointercancel", releaseGravity);
        }

        function captureGravityItem(element, index) {
            // Text-only block elements (headings, paragraphs, status lines) are block-level and
            // span the full container width by default. Shrink-wrap them to their rendered text
            // before measuring so the falling object hugs the text, not a kilometre of empty space.
            const TEXT_BLOCK_TAGS = new Set(["H1", "H2", "H3", "P"]);
            const isTextBlock = TEXT_BLOCK_TAGS.has(element.tagName) ||
                                element.classList.contains("subtitle") ||
                                element.classList.contains("status");

            // Save the original inline width BEFORE any modification so restore is accurate.
            const originalInlineWidth = element.style.width;
            if (isTextBlock) {
                element.style.width = "fit-content";
            }

            const rect = element.getBoundingClientRect();
            const computed = window.getComputedStyle(element);
            const placeholder = document.createElement("div");
            placeholder.style.width = `${rect.width}px`;
            placeholder.style.height = `${rect.height}px`;
            placeholder.style.margin = computed.margin;
            placeholder.style.pointerEvents = "none";
            placeholder.style.visibility = "hidden";
            element.after(placeholder);

            const storedStyle = {
                position: element.style.position,
                left: element.style.left,
                top: element.style.top,
                width: originalInlineWidth, // pre-shrink-wrap value so restore is correct
                height: element.style.height,
                margin: element.style.margin,
                zIndex: element.style.zIndex,
                transform: element.style.transform,
                transformOrigin: element.style.transformOrigin,
                willChange: element.style.willChange,
                cursor: element.style.cursor,
                touchAction: element.style.touchAction,
                userSelect: element.style.userSelect,
            };

            element.style.position = "fixed";
            element.style.left = `${rect.left}px`;
            element.style.top = `${rect.top}px`;
            element.style.width = `${rect.width}px`;
            element.style.height = `${rect.height}px`;
            element.style.margin = "0";
            element.style.zIndex = `${12000 + index}`;
            element.style.transformOrigin = "50% 50%";
            element.style.willChange = "transform";

            document.body.appendChild(element);

            const item = {
                element,
                placeholder,
                storedStyle,
                x: rect.left,
                y: rect.top,
                width: rect.width,
                height: rect.height,
                vx: (Math.random() - 0.5) * 180,
                vy: -60 - Math.random() * 80,
                rotation: 0,
                angularVelocity: (Math.random() - 0.5) * 4,
                dragging: false,
            };

            gravityItems.push(item);
            makeGravityItemInteractive(item);
        }

        function restoreGravityItem(item) {
            item.placeholder.before(item.element);
            Object.entries(item.storedStyle).forEach(([property, value]) => {
                item.element.style[property] = value;
            });
            item.placeholder.remove();
        }

        // Resolve AABB overlap with slop tolerance and partial correction to prevent jitter.
        //
        // Why slop? Without it, every frame the resolver pushes items apart and gravity pulls
        // them back into contact, creating a high-frequency oscillation. Ignoring penetrations
        // smaller than SLOP_PX lets items rest in stable contact without being corrected every frame.
        //
        // Why partial correction? Fixing 100% of the penetration in one frame often over-shoots
        // (items are pushed past separation, then pulled back next frame). CORRECT_FRAC < 1 gives
        // a smooth convergence instead of an oscillation.
        //
        // Why approach guard? Only exchange velocity when the objects are actually closing in on
        // each other. If they are already separating (post-bounce), applying another impulse
        // reverses the separation and causes jitter.
        function resolveCollision(a, b) {
            if (a.dragging || b.dragging) return;

            const overlapX = (a.x + a.width / 2) - (b.x + b.width / 2);
            const overlapY = (a.y + a.height / 2) - (b.y + b.height / 2);
            const halfW = (a.width + b.width) / 2;
            const halfH = (a.height + b.height) / 2;

            const dx = halfW - Math.abs(overlapX);
            const dy = halfH - Math.abs(overlapY);

            if (dx <= 0 || dy <= 0) return; // No overlap

            const SLOP_PX    = 2;    // Penetration depth (px) to tolerate without correcting
            const CORRECT_FRAC = 0.65; // Fraction of remaining overlap resolved per frame
            const RESTITUTION  = 0.05; // Near-zero: objects separate without bouncing

            if (dx < dy) {
                // Horizontal separation axis
                const correction = Math.max(dx - SLOP_PX, 0) * CORRECT_FRAC;
                if (correction === 0) return;
                const sign = overlapX > 0 ? 1 : -1;
                a.x += sign * correction / 2;
                b.x -= sign * correction / 2;
                // Velocity impulse only when closing (approach guard)
                const relVx = a.vx - b.vx;
                if (sign * relVx < 0) { // objects moving toward each other along X
                    const impulse = -(1 + RESTITUTION) * relVx / 2;
                    a.vx += impulse;
                    b.vx -= impulse;
                }
            } else {
                // Vertical separation axis
                const correction = Math.max(dy - SLOP_PX, 0) * CORRECT_FRAC;
                if (correction === 0) return;
                const sign = overlapY > 0 ? 1 : -1;
                a.y += sign * correction / 2;
                b.y -= sign * correction / 2;
                const relVy = a.vy - b.vy;
                if (sign * relVy < 0) { // objects moving toward each other along Y
                    const impulse = -(1 + RESTITUTION) * relVy / 2;
                    a.vy += impulse;
                    b.vy -= impulse;
                }
            }
        }

        function startGravityAnimation() {
            if (gravityAnimation !== null) {
                return;
            }

            let lastTime = performance.now();

            function tick(now) {
                const dt = Math.min(32, now - lastTime) / 1000;
                lastTime = now;
                const floor = window.innerHeight - 8;
                const wallLeft = 8;
                const wallRight = window.innerWidth - 8;

                gravityItems.forEach((item) => {
                    if (item.dragging) {
                        // While dragged, apply position directly; skip physics
                        item.rotation += item.angularVelocity * dt;
                        item.element.style.transform = `translate(${item.x - parseFloat(item.element.style.left)}px, ${item.y - parseFloat(item.element.style.top)}px) rotate(${item.rotation}deg)`;
                        return;
                    }

                    // Gravity
                    item.vy += 1500 * dt;

                    // Air damping
                    item.vx *= Math.pow(0.988, dt * 60);
                    item.vy *= Math.pow(0.995, dt * 60);
                    // Aggressive angular damping — items should barely spin
                    item.angularVelocity *= Math.pow(0.88, dt * 60);
                    // Hard cap: no more than 12 deg/s spin at any time
                    item.angularVelocity = Math.max(-12, Math.min(12, item.angularVelocity));

                    item.x += item.vx * dt;
                    item.y += item.vy * dt;
                    item.rotation += item.angularVelocity * dt;

                    // Left wall
                    if (item.x < wallLeft) {
                        item.x = wallLeft;
                        item.vx = Math.abs(item.vx) * 0.58;
                        // No angular kick on walls — keeps rotation calm
                    }

                    // Right wall
                    if (item.x + item.width > wallRight) {
                        item.x = wallRight - item.width;
                        item.vx = -Math.abs(item.vx) * 0.58;
                    }

                    // Floor
                    if (item.y + item.height > floor) {
                        item.y = floor - item.height;
                        const bounceVy = -Math.abs(item.vy) * 0.32;
                        item.vy = Math.abs(bounceVy) < 28 ? 0 : bounceVy;
                        item.vx *= 0.82;
                        item.angularVelocity *= 0.50; // Spin halves on each floor hit
                    }

                    // Zero out micro-velocities so items can truly come to rest without jittering.
                    // This prevents sub-pixel velocity accumulations from causing endless tiny corrections.
                    if (Math.abs(item.vx) < 2)  item.vx = 0;
                    if (Math.abs(item.vy) < 2)  item.vy = 0;
                    if (Math.abs(item.angularVelocity) < 0.3) item.angularVelocity = 0;

                    item.element.style.transform = `translate(${item.x - parseFloat(item.element.style.left)}px, ${item.y - parseFloat(item.element.style.top)}px) rotate(${item.rotation}deg)`;
                });

                // Inter-element collisions (brute force O(n²) — n is small, typically 4–6 panels)
                for (let i = 0; i < gravityItems.length; i++) {
                    for (let j = i + 1; j < gravityItems.length; j++) {
                        resolveCollision(gravityItems[i], gravityItems[j]);
                    }
                }

                gravityAnimation = window.requestAnimationFrame(tick);
            }

            gravityAnimation = window.requestAnimationFrame(tick);
        }

        function setGravityMode(enabled) {
            gravityMode = enabled;

            if (!enabled) {
                if (gravityAnimation !== null) {
                    window.cancelAnimationFrame(gravityAnimation);
                    gravityAnimation = null;
                }

                while (gravityItems.length) {
                    restoreGravityItem(gravityItems.pop());
                }

                updateModeButtons();
                return;
            }

            gravityTargets().forEach((element, index) => captureGravityItem(element, index));
            startGravityAnimation();
            updateModeButtons();
        }

        function toggleGravityMode() {
            setGravityMode(!gravityMode);
        }

        function bind() {
            elements.pastizziModeButton.addEventListener("click", () => launchPastizzi());
            elements.fireworksButton.addEventListener("click", () => launchFireworks(5 + Math.floor(Math.random() * 4))); // 5-8 fireworks per click; change 5/4 for batch size range.
            elements.partyModeButton.addEventListener("click", togglePartyMode);
            elements.gravityModeButton.addEventListener("click", toggleGravityMode);
            updateModeButtons();
        }

        return {
            bind,
            updateModeButtons,
            launchPastizzi,
            launchFireworks,
            togglePartyMode,
            setPartyMode,
            toggleGravityMode,
            setGravityMode,
        };
    }

    window.createDevToy = createDevToy;
})();
