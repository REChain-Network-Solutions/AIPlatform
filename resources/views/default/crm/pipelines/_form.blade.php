@if ($errors->any())
    <div class="mb-4 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
        <ul class="list-inside list-disc space-y-1">
            @foreach ($errors->all() as $error)<li>{{ $error }}</li>@endforeach
        </ul>
    </div>
@endif

<div class="rounded-2xl border border-border bg-card p-6 shadow-xs">
    <h3 class="mb-5 text-sm font-semibold text-heading-foreground">{{ __('Pipeline Details') }}</h3>
    <div class="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <x-form.group label="{{ __('Pipeline Name') }}" class="sm:col-span-2" required>
            <x-form.input name="name" value="{{ old('name', $pipeline->name ?? '') }}" placeholder="{{ __('e.g. Sales Pipeline') }}" required />
        </x-form.group>
        <x-form.group label="{{ __('Currency') }}">
            <x-form.input name="currency" value="{{ old('currency', $pipeline->currency ?? 'USD') }}" maxlength="3" placeholder="USD" />
        </x-form.group>
    </div>
    <div class="mt-4 flex items-center gap-2">
        <input type="checkbox" name="is_default" id="is_default" value="1" class="lqd-checkbox"
            {{ old('is_default', $pipeline->is_default ?? false) ? 'checked' : '' }}>
        <label for="is_default" class="text-sm text-muted-foreground">{{ __('Set as default pipeline') }}</label>
    </div>
</div>

<div class="mt-5 rounded-2xl border border-border bg-card p-6 shadow-xs">
    <div class="mb-5 flex items-center justify-between">
        <h3 class="text-sm font-semibold text-heading-foreground">{{ __('Stages') }}</h3>
        <button type="button" id="add-stage" class="flex items-center gap-1 rounded-lg px-3 py-1.5 text-xs font-medium text-primary hover:bg-primary/5">
            <x-tabler-plus class="w-3.5" /> {{ __('Add Stage') }}
        </button>
    </div>

    <div id="stages-list" class="space-y-3">
        @php $stages = old('stages', isset($pipeline) ? $pipeline->stages->map(fn($s) => ['id' => $s->id, 'name' => $s->name, 'color' => $s->color, 'probability' => $s->probability, 'is_won' => $s->is_won, 'is_lost' => $s->is_lost])->toArray() : [['name'=>'Qualified','color'=>'#6366f1','probability'=>20,'is_won'=>false,'is_lost'=>false],['name'=>'Proposal','color'=>'#f59e0b','probability'=>50,'is_won'=>false,'is_lost'=>false],['name'=>'Won','color'=>'#22c55e','probability'=>100,'is_won'=>true,'is_lost'=>false],['name'=>'Lost','color'=>'#ef4444','probability'=>0,'is_won'=>false,'is_lost'=>true]]); @endphp

        @foreach ($stages as $i => $stage)
            <div class="stage-row flex items-center gap-3 rounded-xl border border-border bg-background p-3">
                <x-tabler-grip-vertical class="w-4 shrink-0 cursor-grab text-muted-foreground/50" />
                @if (!empty($stage['id']))
                    <input type="hidden" name="stages[{{ $i }}][id]" value="{{ $stage['id'] }}">
                @endif
                <input type="color" name="stages[{{ $i }}][color]" value="{{ $stage['color'] ?? '#6366f1' }}" class="h-8 w-8 shrink-0 cursor-pointer rounded-lg border-0 p-0.5" title="{{ __('Stage colour') }}">
                <input type="text" name="stages[{{ $i }}][name]" value="{{ $stage['name'] ?? '' }}" placeholder="{{ __('Stage name') }}" required class="lqd-input lqd-input-sm flex-1 min-w-0">
                <div class="flex items-center gap-1 text-xs text-muted-foreground shrink-0">
                    <input type="number" name="stages[{{ $i }}][probability]" value="{{ $stage['probability'] ?? 0 }}" min="0" max="100" class="lqd-input lqd-input-sm w-14 text-center"> %
                </div>
                <label class="flex items-center gap-1 text-xs text-muted-foreground shrink-0 cursor-pointer" title="{{ __('Won stage') }}">
                    <input type="checkbox" name="stages[{{ $i }}][is_won]" value="1" class="lqd-checkbox" {{ !empty($stage['is_won']) ? 'checked' : '' }}>
                    {{ __('Won') }}
                </label>
                <label class="flex items-center gap-1 text-xs text-muted-foreground shrink-0 cursor-pointer" title="{{ __('Lost stage') }}">
                    <input type="checkbox" name="stages[{{ $i }}][is_lost]" value="1" class="lqd-checkbox" {{ !empty($stage['is_lost']) ? 'checked' : '' }}>
                    {{ __('Lost') }}
                </label>
                <button type="button" class="remove-stage shrink-0 text-muted-foreground hover:text-red-500" title="{{ __('Remove') }}">
                    <x-tabler-x class="w-4" />
                </button>
            </div>
        @endforeach
    </div>

    <p class="mt-3 text-xs text-muted-foreground">{{ __('Drag to reorder stages. Mark Won/Lost stages to auto-update deal status.') }}</p>
</div>
