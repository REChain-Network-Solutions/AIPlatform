@php
	$items = app(\App\Services\Common\MenuService::class)->generate();

	$user = auth()->user();
	$isAdmin = $user?->isAdmin();

@endphp

@foreach ($items as $key => $item)
	@php
		// Cache values once
		$isActive       = data_get($item, 'is_active', false);
		$showCondition  = data_get($item, 'show_condition', true);
		$isAdminOnly    = data_get($item, 'is_admin', false);
		$childrenCount  = data_get($item, 'children_count', 0);
		$type           = data_get($item, 'type');

		// Skip early if plan doesn’t allow it
		if (!\App\Helpers\Classes\PlanHelper::planMenuCheck($userPlan, $key)) {
			continue;
		}

		// Skip if inactive or condition fails
		if (!$isActive || !$showCondition) {
			continue;
		}

		// Skip if admin-only and user not allowed
		if ($isAdminOnly && (!$isAdmin || !$user->checkPermission($key))) {
			continue;
		}
	@endphp

	{{-- Render item --}}
	@if ($childrenCount)
		@includeIf('default.components.navbar.partials.types.item-dropdown')
	@else
		@includeIf('default.components.navbar.partials.types.' . $type)
	@endif
@endforeach
