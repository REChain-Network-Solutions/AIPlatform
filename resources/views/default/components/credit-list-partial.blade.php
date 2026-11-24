@php
	$isSharedCreditPlan = isset($plan) && $plan->isSharedCreditPlan()
		&& (bool) setting('shared_credit_system_enabled');

	$isSharedCreditUser = !isset($plan) && auth()->check()
		&& auth()->user()->isSharedCreditUser()
		&& (bool) setting('shared_credit_system_enabled');
@endphp

@if ($isSharedCreditPlan)
	<div class="flex items-center justify-between rounded-xl bg-foreground/[3%] px-5 py-4">
		<div class="flex items-center gap-3">
			<x-tabler-bolt class="size-5 text-primary" />
			<span class="font-medium text-heading-foreground">{{ __('Total Credits') }}</span>
		</div>
		<span class="text-lg font-semibold text-heading-foreground">
			@formatNumber($plan->shared_credits_amount)
		</span>
	</div>
@elseif ($isSharedCreditUser)
	@php
		$sharedBalance = (float) auth()->user()->shared_credits;
		$sharedTeam = auth()->user()->myTeam;
		if ($sharedTeam && (!$sharedTeam->exists || !$sharedTeam->isSharedCreditTeam())) {
			$sharedTeam = null;
		}
	@endphp

	<div class="flex flex-col gap-4">
		<div class="flex items-center justify-between rounded-xl bg-foreground/[3%] px-5 py-4">
			<div class="flex items-center gap-3">
				<x-tabler-bolt class="size-5 text-primary" />
				<span class="font-medium text-heading-foreground">{{ __('Your Credits') }}</span>
			</div>
			<span class="text-lg font-semibold text-heading-foreground">
				@formatNumber($sharedBalance)
			</span>
		</div>

		@if ($sharedTeam)
			<div class="flex items-center justify-between rounded-xl bg-foreground/[3%] px-5 py-4">
				<div class="flex items-center gap-3">
					<x-tabler-users class="size-5 text-secondary" />
					<span class="font-medium text-heading-foreground">{{ __('Team Pool') }}</span>
				</div>
				<span class="text-lg font-semibold text-heading-foreground">
					@formatNumber($sharedTeam->shared_credits)
				</span>
			</div>

			<div class="flex items-center justify-between rounded-xl bg-foreground/[3%] px-5 py-4">
				<div class="flex items-center gap-3">
					<x-tabler-calculator class="size-5 text-foreground/50" />
					<span class="font-medium text-heading-foreground">{{ __('Total Available') }}</span>
				</div>
				<span class="text-lg font-semibold text-primary">
					@formatNumber($sharedBalance + $sharedTeam->shared_credits)
				</span>
			</div>
		@endif
	</div>
@else
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

			@includeIf('video-dubbing::partials.credit-list-row', [
				'plan' => $plan ?? null,
				'user' => $user ?? null,
			])

			@includeIf('ugc-creator::partials.credit-list-row', [
				'plan' => $plan ?? null,
				'user' => $user ?? null,
			])

			@includeIf('ai-captions::partials.credit-list-row', [
				'plan' => $plan ?? null,
				'user' => $user ?? null,
			])

		</tbody>
	</table>
@endif
