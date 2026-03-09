<table class="mb-4 w-full table-auto border-collapse border">
    <thead>
        <tr class="bg-foreground/10">
            <th class="border p-2 text-start">{{ __('Model') }}</th>
            <th class="border p-2 text-end">{{ __('Credits') }}</th>
        </tr>
    </thead>
    <tbody>
        @foreach ($categories as $key => $model)
			@php
				if (isset($plan) && $plan->exists) {
					$drivers = $model->forPlan($plan)->list();
				} else {
					$drivers = $model->forUser($user ?? auth()->user())->list();
				}

				$groupName = $drivers->isNotEmpty() ? $drivers->first()->enum()->subLabel() : '';
				$isUnlimited = $model->checkIfThereUnlimited();
				$credits = $model->totalCredits();
				$tooltip_anchor = $loop->index < 4 ? 'top' : 'bottom';
			@endphp

			@if (!$isUnlimited && $credits <= 0)
                @continue
            @endif
            <tr>
                <td class="flex justify-between border-b p-2">
                    {{ $groupName }}
                    <x-info-tooltip
                        class:content="max-h-48 overflow-y-auto"
                        :drivers="$drivers"
                        :anchor="$tooltip_anchor"
                    />
                </td>
                <td class="border p-2 text-end">
                    {{ $isUnlimited ? __('Unlimited') : $credits }}
                </td>
            </tr>
        @endforeach
    </tbody>
</table>
