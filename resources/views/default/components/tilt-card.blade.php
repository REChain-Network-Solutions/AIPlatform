@props([
    'intensity' => 15,
    'perspective' => 1000,
    'scale' => 1.02,
    'speed' => 400,
    'glare' => true,
    'glareOpacity' => 0.2,
])

@php
    $base_class = 'lqd-tilt-card relative';
@endphp

<div
    {{ $attributes->withoutTwMergeClasses()->twMerge($base_class, $attributes->get('class')) }}
    x-data="lqdTiltCard({
        intensity: {{ $intensity }},
        perspective: {{ $perspective }},
        scale: {{ $scale }},
        speed: {{ $speed }},
        glare: {{ $glare ? 'true' : 'false' }},
        glareOpacity: {{ $glareOpacity }}
    })"
    x-on:mouseenter="onMouseEnter"
    x-on:mousemove="onMouseMove"
    x-on:mouseleave="onMouseLeave"
    x-ref="card"
    :style="cardStyle"
>
    <div
        class="lqd-tilt-card-content relative min-h-full min-w-full"
        x-ref="content"
        :style="contentStyle"
    >
        {{ $slot }}
    </div>

    @if ($glare)
        <div
            class="lqd-tilt-card-glare pointer-events-none absolute inset-0 overflow-hidden rounded-[inherit]"
            x-ref="glareWrapper"
            :style="{ opacity: isHovering ? 1 : 0, transition: `opacity ${speed}ms ease-out` }"
        >
            <div
                class="bg-gradient-radial absolute -inset-[50%] from-white/50 to-transparent"
                x-ref="glare"
                :style="glareStyle"
            ></div>
        </div>
    @endif
</div>

@pushOnce('script')
    <script>
        document.addEventListener('alpine:init', () => {
            Alpine.data('lqdTiltCard', (config) => ({
                intensity: config.intensity ?? 15,
                perspective: config.perspective ?? 1000,
                scale: config.scale ?? 1.02,
                speed: config.speed ?? 400,
                glare: config.glare ?? true,
                glareOpacity: config.glareOpacity ?? 0.2,

                isHovering: false,
                rotateX: 0,
                rotateY: 0,
                glareX: 50,
                glareY: 50,

                get isTouchDevice() {
                    return window.matchMedia('(pointer: coarse)').matches ||
                        'ontouchstart' in window ||
                        navigator.maxTouchPoints > 0;
                },

                get cardStyle() {
                    return {
                        perspective: `${this.perspective}px`,
                        transformStyle: 'preserve-3d'
                    };
                },

                get contentStyle() {
                    const transform = this.isHovering ?
                        `rotateX(${this.rotateX}deg) rotateY(${this.rotateY}deg) scale3d(${this.scale}, ${this.scale}, ${this.scale})` :
                        'rotateX(0deg) rotateY(0deg) scale3d(1, 1, 1)';

                    return {
                        transform,
                        transformStyle: 'preserve-3d',
                        transition: `transform ${this.speed}ms ease-out`
                    };
                },

                get glareStyle() {
                    return {
                        transform: `translate(${this.glareX - 50}%, ${this.glareY - 50}%)`,
                        opacity: this.glareOpacity,
                        transition: `transform ${this.speed}ms ease-out`
                    };
                },

                onMouseEnter() {
                    if (this.isTouchDevice) return;
                    this.isHovering = true;
                },

                onMouseMove(event) {
                    if (this.isTouchDevice || !this.isHovering) return;

                    const card = this.$refs.card;
                    const rect = card.getBoundingClientRect();

                    const x = event.clientX - rect.left;
                    const y = event.clientY - rect.top;

                    const centerX = rect.width / 2;
                    const centerY = rect.height / 2;

                    const percentX = (x - centerX) / centerX;
                    const percentY = (y - centerY) / centerY;

                    this.rotateX = -percentY * this.intensity;
                    this.rotateY = percentX * this.intensity;

                    this.glareX = (x / rect.width) * 100;
                    this.glareY = (y / rect.height) * 100;

                    this.updateDepthElements(percentX, percentY);
                },

                onMouseLeave() {
                    if (this.isTouchDevice) return;
                    this.isHovering = false;
                    this.rotateX = 0;
                    this.rotateY = 0;
                    this.glareX = 50;
                    this.glareY = 50;

                    this.resetDepthElements();
                },

                updateDepthElements(percentX, percentY) {
                    const elements = this.$refs.content.querySelectorAll('[data-depth]');

                    elements.forEach(el => {
                        const depth = parseFloat(el.dataset.depth) || 0;
                        const translateX = percentX * depth * 20;
                        const translateY = percentY * depth * 20;
                        const translateZ = depth * 30;

                        el.style.transform = `translate3d(${translateX}px, ${translateY}px, ${translateZ}px)`;
                        el.style.transition = `transform ${this.speed}ms ease-out`;
                    });
                },

                resetDepthElements() {
                    const elements = this.$refs.content.querySelectorAll('[data-depth]');

                    elements.forEach(el => {
                        el.style.transform = 'translate3d(0, 0, 0)';
                    });
                }
            }));
        });
    </script>
@endPushOnce

@pushOnce('style')
    <style>
        .bg-gradient-radial {
            background: radial-gradient(circle at center, var(--tw-gradient-from) 0%, var(--tw-gradient-to) 70%);
        }
    </style>
@endPushOnce
