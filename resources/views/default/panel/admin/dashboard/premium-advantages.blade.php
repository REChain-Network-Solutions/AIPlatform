@if ((! $vip_membership) && $app_is_not_demo)
    <x-card
        class="relative flex items-center border-4"
        class:body="static rounded-[inherit] only:grow-0 lg:p-10 w-full"
        id="{{ 'admin-card-' . ($widget?->name?->value ?? 'premium-advantages') }}"
    >
        <x-outline-glow
            class="[--glow-color-primary:238deg_71%_79%] [--glow-color-secondary:166deg_74%_45%] [--outline-glow-iteration:2] [--outline-glow-w:4px]"
            effect="3"
        />

        <div class="absolute end-8 mb-6 inline-grid size-14 place-content-center rounded-xl bg-gradient-to-br from-gradient-from via-gradient-via to-gradient-to text-white">
            <x-tabler-diamond
                class="size-10"
                stroke-width="1.5"
            />
        </div>
        <h3 class="mb-6">
            @lang('Premium Advantages')
        </h3>
        <ul class="mb-11 space-y-4 self-center text-xs font-medium">
            @foreach ($premium_features as $feature => $tooltip)
                <li class="flex items-center gap-3.5">
                    <svg
                        width="16"
                        height="16"
                        viewBox="0 0 16 16"
                        fill="none"
                        xmlns="http://www.w3.org/2000/svg"
                    >
                        <path
                            d="M2.09635 7.37072C1.80296 7.37154 1.51579 7.45542 1.26807 7.61264C1.02035 7.76986 0.822208 7.994 0.696564 8.25914C0.570919 8.52427 0.522908 8.81956 0.558084 9.11084C0.59326 9.40212 0.710186 9.67749 0.895335 9.9051L4.84228 14.7401C4.98301 14.9148 5.1634 15.0535 5.36847 15.1445C5.57353 15.2355 5.79736 15.2763 6.02136 15.2635C6.50043 15.2377 6.93295 14.9815 7.20871 14.5601L15.4075 1.35593C15.4089 1.35373 15.4103 1.35154 15.4117 1.34939C15.4886 1.23127 15.4637 0.997192 15.3049 0.850142C15.2613 0.809761 15.2099 0.778736 15.1538 0.75898C15.0977 0.739223 15.0382 0.731153 14.9789 0.735266C14.9196 0.739379 14.8618 0.755589 14.809 0.782896C14.7562 0.810204 14.7095 0.848031 14.6719 0.894048C14.669 0.897666 14.6659 0.90123 14.6628 0.904739L6.39421 10.247C6.36275 10.2826 6.32454 10.3115 6.28179 10.3322C6.23905 10.3528 6.19263 10.3648 6.14522 10.3674C6.09782 10.3699 6.05038 10.363 6.00565 10.3471C5.96093 10.3312 5.91982 10.3065 5.88471 10.2746L3.14051 7.77735C2.8555 7.51608 2.48299 7.37102 2.09635 7.37072Z"
                            fill="url(#paint0_linear_9208_560_{{ $loop->index }})"
                        />
                        <defs>
                            <linearGradient
                                id="paint0_linear_9208_560_{{ $loop->index }}"
                                x1="0.546875"
                                y1="3.69866"
                                x2="12.7738"
                                y2="14.7613"
                                gradientUnits="userSpaceOnUse"
                            >
                                <stop stop-color="hsl(var(--gradient-from))" />
                                <stop
                                    offset="0.502"
                                    stop-color="hsl(var(--gradient-via))"
                                />
                                <stop
                                    offset="1"
                                    stop-color="hsl(var(--gradient-to))"
                                />
                            </linearGradient>
                        </defs>
                    </svg>
                    {!! $feature !!} <x-info-tooltip :text="$tooltip" />
                </li>
            @endforeach
        </ul>

        <x-button
            class="w-full shadow-md shadow-black/[7%] hover:bg-primary hover:text-primary-foreground hover:shadow-xl hover:shadow-primary/20 hover:outline-primary"
            href="/subscription"
            variant="outline"
        >
            @lang('Join Premium')
        </x-button>
    </x-card>
@endif
