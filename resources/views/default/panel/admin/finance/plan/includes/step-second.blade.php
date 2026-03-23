<div class="space-y-8">

    @if ($planAiToolsMenu)
        <div class="grid grid-cols-1 gap-8 sm:grid-cols-2">
            <x-form-step
                class="col-span-2 m-0"
                step="1"
                label="{{ __('AI Tools') }}"
            />
            @foreach ($planAiToolsMenu as $tool)
                <x-form.group
                    class="col-span-2 sm:col-span-1"
                    no-group-label
                    :error="'plan.plan_ai_tools.' . $tool['key']"
                >
                    <x-form.checkbox
                        class="border-input rounded-input border !px-2.5 !py-3"
                        wire:model="plan.plan_ai_tools.{{ $tool['key'] }}"
                        value="{{ $tool['key'] }}"
                        label="{{ $tool['label'] }}"
                        tooltip="{{ $tool['tooltip'] ?? $tool['label'] }}"
                    />
                </x-form.group>

                @if ($tool['key'] === \App\Enums\Introduction::AI_EXT_VOICE_CALL->value)
                    <x-form.group
                        class="col-span-2 sm:col-span-1"
                        label="{{ __('Voice Call Seconds Limit') }}"
                        tooltip="{{ __('-1 for unlimited, 0 to disable, >0 for max seconds per month') }}"
                        error="plan.voice_call_seconds_limit"
                    >
                        <x-form.stepper
                            wire:model="plan.voice_call_seconds_limit"
                            step="1"
                            min="-1"
                        />
                    </x-form.group>
                @endif
            @endforeach
        </div>
    @endif

    <div class="grid grid-cols-1 gap-8 sm:grid-cols-2">
        <x-form-step
            class="col-span-2 m-0"
            step="2"
            label="{{ __('Features') }}"
        />
        @foreach ($planFeatureMenu as $feature)
            <x-form.group
                class="col-span-2 sm:col-span-1"
                no-group-label
                :error="'plan.plan_features.' . $feature['key']"
            >
                <x-form.checkbox
                    class="border-input rounded-input border !px-2.5 !py-3"
                    wire:model="plan.plan_features.{{ $feature['key'] }}"
                    label="{{ $feature['label'] }}"
                    value="{{ $feature['key'] }}"
                />
            </x-form.group>
        @endforeach
    </div>
</div>
