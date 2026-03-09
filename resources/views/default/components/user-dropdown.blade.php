@php
	$user = auth()?->user();
	$user_avatar = $user?->avatar;

	if (!$user?->github_token && !$user?->google_token && !$user?->facebook_token) {
	   $user_avatar = '/' . $user_avatar;
	}

	$avatar_url = custom_theme_url($user_avatar);
@endphp

<x-dropdown.dropdown
	{{ $attributes->twMerge('header-user-dropdown') }}
	anchor="end"
	offsetY="{{ $attributes->get('dropdown-offset-y') ?? '20px' }}"
>
	<x-slot:trigger
		class="{{ @twMerge('size-9 p-0', $attributes->get('class:trigger')) }}"
	>
		@if (isset($trigger) && filled($trigger))
			{{ $trigger }}
		@else
			<span
				class="inline-block size-full rounded-full bg-cover bg-center"
				style="background-image: url({{ $avatar_url }})"
				role="img"
				aria-label="{{ __('User avatar') }}"
			></span>
		@endif
	</x-slot:trigger>

	<x-slot:dropdown
		class="w-52"
	>
		<div class="px-3 pt-3">
			<p class="m-0 text-foreground">{{ $user->fullName() }}</p>
			<p class="text-3xs text-foreground/70">{{ $user->email }}</p>
		</div>

		<hr>

		<x-credit-list
			class:legends="gap-1"
			class:modal-trigger="text-2xs w-full"
			modal-trigger-variant="ghost-shadow"
			modal-trigger-pos="block"
			expanded-modal-trigger
		/>

		<hr>

		<div class="pb-2 text-2xs">
			@foreach([
			   ['route' => 'dashboard.user.2fa.activate', 'label' => '2-Factor Auth.'],
			   ['route' => 'dashboard.user.payment.subscription', 'label' => 'Plan'],
			   ['route' => 'dashboard.user.orders.index', 'label' => 'Orders'],
			   ['route' => 'dashboard.user.settings.index', 'label' => 'Settings'],
			] as $item)

				<a class="flex w-full items-center px-3 py-2 hover:bg-foreground/5"
				   href="{{ route($item['route']) }}"
				>
					{{ __($item['label']) }}
				</a>
			@endforeach

			<form
				class="flex w-full"
				id="logout"
				method="POST"
				action="{{ route('logout') }}"
			>
				@csrf
				<button
					class="flex w-full items-center px-3 py-2 hover:bg-foreground/10"
					type="submit"
				>
					{{ __('Logout') }}
				</button>
			</form>
		</div>

	</x-slot:dropdown>
</x-dropdown.dropdown>