@if($plan->isSharedCreditPlan())
	<div class="rounded-xl border border-border p-6 text-center">
		<x-tabler-coins class="mx-auto mb-3 size-12 text-primary" />
		<h4 class="mb-2">{{ __('Shared Credit Plan') }}</h4>
		<p class="mb-3 text-foreground/60">
			{{ __('This plan uses a shared credit pool. Individual model credits are not needed.') }}
		</p>
		<div class="text-3xl font-bold text-primary">
			{{ number_format($plan->shared_credits_amount ?? 0) }}
		</div>
		<span class="text-foreground/60">{{ __('shared credits') }}</span>
	</div>
@else
	@livewire('assign-view-credits', ['entities' => $plan->ai_models, 'plan' => $plan])
@endif
