@php
	use App\Domains\Entity\Enums\EntityEnum;

	$base_class = 'lqd-remaining-credit relative mx-2 flex flex-col gap-3 text-2xs';
	$progress_base_class = 'lqd-progress flex h-2 overflow-hidden rounded-full';
	$progressbar_text_base_class = 'lqd-progress-bar grow-0 basis-auto bg-primary';
	$progressbar_image_base_class = 'lqd-progress-bar grow-0 basis-auto bg-secondary';
	$legend_text_base_class = 'group';
	$legend_box_text_base_class = '';
	$legend_image_base_class = 'group';
	$legend_box_image_base_class = 'bg-secondary';
	$modal_trigger_base_class = '';

	$variations = [
		'progressHeight' => [
			'sm' => 'h-1',
			'md' => 'h-2',
		],
	];

	$progressHeight = $variations['progressHeight'][$progressHeight] ?? $variations['progressHeight']['md'];

	$random = random_int(100000, 900000);

	if ($modalTriggerPos === 'inline' && $showType !== 'button') {
		$base_class .= ' pe-12';
		$modal_trigger_base_class .= ' absolute end-0 top-0 size-9 shrink-0 p-0 outline-heading-foreground/10 hover:bg-primary hover:text-primary-foreground';
	}
@endphp

@if ($showType === 'directly')
	<div id="credit-list-partial-direct-{{ $random }}">
		<div class="grid min-h-[140px] w-full place-items-center overflow-x-scroll rounded-lg p-6 lg:overflow-visible">
			<svg
				class="animate-spin text-gray-300"
				viewBox="0 0 64 64"
				fill="none"
				xmlns="http://www.w3.org/2000/svg"
				width="24"
				height="24"
			>
				<path
					d="M32 3C35.8083 3 39.5794 3.75011 43.0978 5.20749C46.6163 6.66488 49.8132 8.80101 52.5061 11.4939C55.199 14.1868 57.3351 17.3837 58.7925 20.9022C60.2499 24.4206 61 28.1917 61 32C61 35.8083 60.2499 39.5794 58.7925 43.0978C57.3351 46.6163 55.199 49.8132 52.5061 52.5061C49.8132 55.199 46.6163 57.3351 43.0978 58.7925C39.5794 60.2499 35.8083 61 32 61C28.1917 61 24.4206 60.2499 20.9022 58.7925C17.3837 57.3351 14.1868 55.199 11.4939 52.5061C8.801 49.8132 6.66487 46.6163 5.20749 43.0978C3.7501 39.5794 3 35.8083 3 32C3 28.1917 3.75011 24.4206 5.2075 20.9022C6.66489 17.3837 8.80101 14.1868 11.4939 11.4939C14.1868 8.80099 17.3838 6.66487 20.9022 5.20749C24.4206 3.7501 28.1917 3 32 3L32 3Z"
					stroke="currentColor"
					stroke-width="5"
					stroke-linecap="round"
					stroke-linejoin="round"
				></path>
				<path
					class="text-gray-900"
					d="M32 3C36.5778 3 41.0906 4.08374 45.1692 6.16256C49.2477 8.24138 52.7762 11.2562 55.466 14.9605C58.1558 18.6647 59.9304 22.9531 60.6448 27.4748C61.3591 31.9965 60.9928 36.6232 59.5759 40.9762"
					stroke="currentColor"
					stroke-width="5"
					stroke-linecap="round"
					stroke-linejoin="round"
				>
				</path>
			</svg>
		</div>
	</div>

	<script>
		fetch('{!! route('credit-list-partial', ['cache_key' => $plan?->id ? 'credit-list-plan-cache' : request('credit-list-cache'), 'plan_id' => $plan?->id, 'user_id' => $user?->id]) !!}').then(response => response.json())
			.then(data => {
				let ID1 = '#credit-list-partial-direct-{{ $random }}';
				let ID2 = '#credit-list-partial-{{ $random }}';

				if (document.querySelector(ID1)) {
					document.querySelector(ID1).innerHTML = data.html;
				}

				if (document.querySelector(ID2)) {
					document.querySelector(ID2).innerHTML = data.html;
				}
			});
	</script>
@else
	@php
		$isSharedCreditUser = auth()->check()
			&& auth()->user()->isSharedCreditUser()
			&& (bool) setting('shared_credit_system_enabled');

		$sharedCreditBalance = $isSharedCreditUser ? (float) auth()->user()->shared_credits : 0;
		$sharedCreditTeam = null;
		if ($isSharedCreditUser) {
			$sharedCreditTeam = $team ?? auth()->user()->myTeam;
			if ($sharedCreditTeam && (!$sharedCreditTeam->exists || !$sharedCreditTeam->isSharedCreditTeam())) {
				$sharedCreditTeam = null;
			}
		}

		if ($showLegend) {
			$wordContainUnlimited = $imageContainUnlimited = false;
			$imageCreditsCount = $wordCreditsCount = 0;
			$wordEntities = $imageEntities = null;

			if (auth()->check()) {
				if (isset($team) && $team?->exists) {
					$wordEntities = \App\Domains\Entity\EntityStats::word()->forTeam($team);
					$imageEntities = \App\Domains\Entity\EntityStats::image()->forTeam($team);
				} else {
					$wordEntities = \App\Domains\Entity\EntityStats::word()->forUser($user);
					$imageEntities = \App\Domains\Entity\EntityStats::image()->forUser($user);
				}

				$wordContainUnlimited = $wordEntities->checkIfThereUnlimited();
				$imageContainUnlimited = $imageEntities->checkIfThereUnlimited();

				$wordCreditsCount = $wordEntities->totalCredits();
				$imageCreditsCount = $imageEntities->totalCredits();
			}
			$totalCreditsCount = $imageCreditsCount + $wordCreditsCount;
			$totalCreditsCount = (int) $totalCreditsCount === 0 ? 1 : $totalCreditsCount;
			if ($wordContainUnlimited && $imageContainUnlimited) {
				$progressbar_text_base_class .= ' shrink-1';
				$progressbar_image_base_class .= ' shrink-1';
			} else {
				$progressbar_text_base_class .= ' shrink-0';
				$progressbar_image_base_class .= ' shrink-0';
			}

        	$uniqueDriversByDefaultImageModel = $imageEntities
            ? $imageEntities
                ->list()
                ->filter(function ($driver) {
                    $engine = $driver->engine();
                    $defaultModel = $engine?->getDefaultImageModel();
                    return $defaultModel && EntityEnum::fromSlug($driver->enum()->slug()) === $defaultModel;
                })
                ->unique(function ($driver) {
                    return $driver->engine()->value;
                })
            : collect();
		}
	@endphp

	<div
		{{ $attributes->withoutTwMergeClasses()->twMerge($base_class, $attributes->get('class')) }}
		@if ($aiImage && $showLegend) x-data="{
			init() {
				if ( this.activeGenerator ) {
					this.generator = this.activeGenerator;
					this.$watch('activeGenerator', value => {
						if ( value === 'flux-pro' ) {
							value = 'fal_ai';
						}
						this.generator = value;
					});
				}
			},
            _generator: '{{ $uniqueDriversByDefaultImageModel->first()?->engine()->value }}',
			get generator() {
				return this._generator;
			},
			set generator(value) {
				this._generator = value;
			}
        }"
		@active-generator-changed.window="console.log($event.detail);generator = $event.detail" @endif
	>
		@if ($showType !== 'button' && $showLegend)
			@if ($isSharedCreditUser)
				<div class="{{ @twMerge($style === 'inline' ? 'lqd-remaining-credits-legends flex items-center justify-between gap-3 gap-y-1.5 flex-wrap' : '', $attributes->get('class:legends')) }}">
					<x-legend
						class="{{ @twMerge($legend_text_base_class, $attributes->get('class:legend-text')) }}"
						class:box="{{ @twMerge('bg-primary', $attributes->get('class:legend-text-box')) }}"
						size="{{ $legendSize }}"
						label="{{ __('Shared Credits') }}"
					>
						<span id="lqd-shared-credit-balance-{{ $random }}" class="ms-auto font-medium">
							@formatNumberShort($sharedCreditBalance)
						</span>
					</x-legend>
					@if ($sharedCreditTeam)
						<x-legend
							class="{{ @twMerge($legend_text_base_class, $attributes->get('class:legend-text')) }}"
							class:box="{{ @twMerge('bg-secondary', $attributes->get('class:legend-text-box')) }}"
							size="{{ $legendSize }}"
							label="{{ __('Team Pool') }}"
						>
							<span class="ms-auto font-medium">
								@formatNumberShort($sharedCreditTeam->shared_credits)
							</span>
						</x-legend>
					@endif
				</div>
				<div {{ $attributes->twMergeFor('progress', $progress_base_class, $progressHeight) }}>
					<div
						{{ $attributes->twMergeFor('progressbar-text', 'lqd-progress-bar grow-0 basis-auto bg-primary shrink-0') }}
						style="width: 100%"
					></div>
				</div>
				<script>
					(function () {
						var ids = {
							badge:      'lqd-shared-credit-balance-{{ $random }}',
							modal:      'lqd-shared-credit-modal-balance-{{ $random }}',
							modalTeam:  'lqd-shared-credit-modal-team-{{ $random }}',
							modalTotal: 'lqd-shared-credit-modal-total-{{ $random }}',
						};
						function lqdFormatNumberShort(n) {
							if (n === null || n === undefined || isNaN(n)) { return '0'; }
							var suffixes = ['', 'K', 'M', 'B', 'T', 'Qa', 'Qi'];
							var i = Math.max(0, Math.min(suffixes.length - 1, Math.floor(Math.log(Math.abs(n) || 1) / Math.log(1000))));
							var short = n / Math.pow(1000, i);
							return parseFloat(short.toFixed(3)).toString() + suffixes[i];
						}
						function lqdFormatNumber(n) {
							if (n === null || n === undefined || isNaN(n)) { return '0'; }
							return new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 }).format(n);
						}
						function setText(id, val) {
							var el = document.getElementById(id);
							if (el) { el.textContent = val; }
						}
						window.addEventListener('lqd:credit-balance-refresh', function () {
							fetch('/shared-credit/balance', {
								headers: {
									'X-CSRF-TOKEN': document.querySelector('meta[name="csrf-token"]')?.content || '',
									'Accept': 'application/json',
								},
								credentials: 'same-origin',
							})
								.then(function (r) { return r.json(); })
								.then(function (data) {
									if (typeof data.balance !== 'number') { return; }
									setText(ids.badge, lqdFormatNumberShort(data.balance));
									setText(ids.modal, lqdFormatNumber(data.balance));
									if (data.team_balance !== null && data.team_balance !== undefined) {
										setText(ids.modalTeam, lqdFormatNumber(data.team_balance));
										setText(ids.modalTotal, lqdFormatNumber(data.balance + data.team_balance));
									}
								})
								.catch(function () {});
						});
					}());
				</script>
			@else
				<div
					class="{{ @twMerge($style === 'inline' ? 'lqd-remaining-credits-legends flex items-center justify-between gap-3 gap-y-1.5 flex-wrap' : '', $attributes->get('class:legends')) }}">
					@if ($aiImage)
						@foreach ($uniqueDriversByDefaultImageModel as $driver)
							<x-legend
								class="{{ @twMerge(['hidden'], $legend_text_base_class, $attributes->get('class:legend-text')) }}"
								class:box="{{ @twMerge($legend_box_text_base_class, $attributes->get('class:legend-text-box')) }}"
								class:label="{{ @twMerge($attributes->get('class:legend-text-label')) }}"
								id="generator-legend-{{ $driver->engine()->value }}"
								size="{{ $legendSize }}"
								label="{{ __($driver->enum()->value) }}"
								::class="{ hidden: generator !== '{{ $driver->engine()->value }}', flex: generator === '{{ $driver->engine()->value }}' }"
							>
								<span class="ms-auto font-medium">
									{{ $driver->isUnlimitedCredit() ? __('Unlimited') : $driver->creditBalance() }}
								</span>
							</x-legend>
						@endforeach
					@else
						<x-legend
							class="{{ @twMerge($legend_text_base_class, $attributes->get('class:legend-text')) }}"
							class:box="{{ @twMerge($legend_box_text_base_class, $attributes->get('class:legend-text-box')) }}"
							class:label="{{ @twMerge($attributes->get('class:legend-text-label')) }}"
							size="{{ $legendSize }}"
							label="{{ __($labelWords) }}"
						>
							<span class="ms-auto font-medium">
								@formatNumberShort($wordEntities?->checkIfThereUnlimited() ? __('Unlimited') : $wordEntities?->totalCredits())
							</span>
							@if (!$wordEntities->checkIfThereUnlimited())
								<span
									class="pointer-events-none invisible absolute bottom-full left-1/2 mb-1 -translate-x-1/2 translate-y-1 scale-90 rounded-md bg-heading-foreground/10 px-2 py-1 font-medium leading-none text-heading-foreground opacity-0 blur-md backdrop-blur-lg transition-all group-hover:visible group-hover:translate-y-0 group-hover:scale-100 group-hover:opacity-100 group-hover:blur-0"
								>
									@formatNumber($wordEntities->totalCredits())
								</span>
							@endif
						</x-legend>
					@endif

					<x-legend
						class="{{ @twMerge($legend_image_base_class, $attributes->get('class:legend-image')) }}"
						class:box="{{ @twMerge($legend_box_image_base_class, $attributes->get('class:legend-image-box')) }}"
						class:label="{{ @twMerge($attributes->get('class:legend-image-label')) }}"
						size="{{ $legendSize }}"
						label="{{ __($labelImages) }}"
					>
						<span class="ms-auto font-medium">
							@formatNumberShort($imageEntities?->checkIfThereUnlimited() ? __('Unlimited') : $imageEntities?->totalCredits())
						</span>
						@if (!$imageEntities->checkIfThereUnlimited())
							<span
								class="pointer-events-none invisible absolute bottom-full left-1/2 mb-1 -translate-x-1/2 translate-y-1 scale-90 rounded-md bg-heading-foreground/10 px-2 py-1 font-medium leading-none text-heading-foreground opacity-0 blur-md backdrop-blur-lg transition-all group-hover:visible group-hover:translate-y-0 group-hover:scale-100 group-hover:opacity-100 group-hover:blur-0"
							>
								@formatNumber($imageEntities->totalCredits())
							</span>
						@endif
					</x-legend>
				</div>
				<div {{ $attributes->twMergeFor('progress', $progress_base_class, $progressHeight) }}>
					<div
						{{ $attributes->twMergeFor('progressbar-text', $progressbar_text_base_class) }}
						style="width: {{ $wordContainUnlimited ? 100 : ($wordCreditsCount / $totalCreditsCount) * 100 }}%"
					></div>
					<div
						{{ $attributes->twMergeFor('progressbar-image', $progressbar_image_base_class) }}
						style="width: {{ $imageContainUnlimited ? 100 : ($imageCreditsCount / $totalCreditsCount) * 100 }}%"
					></div>
				</div>
			@endif
		@endif

		<x-modal
			@class([
				@twMerge(
					'static',
					$modalTriggerPos === 'inline' ? '-mt-3' : '',
					$attributes->get('class:modal')),
			])
			title="{{ $isSharedCreditUser ? __('Your Shared Credits') : __('Your Credit List') }}"
			disable-focus
		>
			<x-slot:trigger
				class="{{ @twMerge($modal_trigger_base_class, $attributes->get('class:modal-trigger')) }}"
				variant="{{ $attributes->has('modal-trigger-variant') ? $attributes->get('modal-trigger-variant') : 'outline' }}"
				disable-focus
				@click="{{ $isSharedCreditUser ? '' : 'loadCredits' . $random . '()' }}"
				title="{{ __('View Your Credits') }}"
			>
				@if (isset($trigger_label))
					{{ $trigger_label }}
				@else
					@if ($attributes->has('expanded-modal-trigger'))
						{{ __('View Your Credits') }}
					@else
						<x-tabler-eye class="size-4"/>
					@endif
				@endif
			</x-slot:trigger>
			<x-slot:modal>
				@if ($isSharedCreditUser)
					<h3 class="mb-2">{{ __('Shared Credit Balance') }}</h3>
					<p class="mb-5 text-foreground/70">{{ __('Your plan uses a single shared credit pool for all AI operations.') }}</p>

					<div class="flex flex-col gap-4">
						<div class="flex items-center justify-between rounded-xl bg-foreground/[3%] px-5 py-4">
							<div class="flex items-center gap-3">
								<x-tabler-bolt class="size-5 text-primary" />
								<span class="font-medium text-heading-foreground">{{ __('Your Credits') }}</span>
							</div>
							<span id="lqd-shared-credit-modal-balance-{{ $random }}" class="text-lg font-semibold text-heading-foreground">
								@formatNumber($sharedCreditBalance)
							</span>
						</div>

						@if ($sharedCreditTeam)
							<div class="flex items-center justify-between rounded-xl bg-foreground/[3%] px-5 py-4">
								<div class="flex items-center gap-3">
									<x-tabler-users class="size-5 text-secondary" />
									<span class="font-medium text-heading-foreground">{{ __('Team Pool') }}</span>
								</div>
								<span id="lqd-shared-credit-modal-team-{{ $random }}" class="text-lg font-semibold text-heading-foreground">
									@formatNumber($sharedCreditTeam->shared_credits)
								</span>
							</div>

							<div class="flex items-center justify-between rounded-xl bg-foreground/[3%] px-5 py-4">
								<div class="flex items-center gap-3">
									<x-tabler-calculator class="size-5 text-foreground/50" />
									<span class="font-medium text-heading-foreground">{{ __('Total Available') }}</span>
								</div>
								<span id="lqd-shared-credit-modal-total-{{ $random }}" class="text-lg font-semibold text-primary">
									@formatNumber($sharedCreditBalance + $sharedCreditTeam->shared_credits)
								</span>
							</div>
						@endif
					</div>
				@else
					<h3 class="mb-2">{{ __('Unlock your creativity with credits') }}</h3>
					<p class="mb-5">{{ __('Each credit unlocks powerful AI tools and features designed to enhance your content creation.') }}</p>

					<div
						class="credit-list-partial"
						id="credit-list-partial-{{ $random }}"
					>
						<div class="grid min-h-[140px] w-full place-items-center overflow-x-scroll rounded-lg p-6 lg:overflow-visible">
							<svg
								class="animate-spin text-gray-300"
								viewBox="0 0 64 64"
								fill="none"
								xmlns="http://www.w3.org/2000/svg"
								width="24"
								height="24"
							>
								<path
									d="M32 3C35.8083 3 39.5794 3.75011 43.0978 5.20749C46.6163 6.66488 49.8132 8.80101 52.5061 11.4939C55.199 14.1868 57.3351 17.3837 58.7925 20.9022C60.2499 24.4206 61 28.1917 61 32C61 35.8083 60.2499 39.5794 58.7925 43.0978C57.3351 46.6163 55.199 49.8132 52.5061 52.5061C49.8132 55.199 46.6163 57.3351 43.0978 58.7925C39.5794 60.2499 35.8083 61 32 61C28.1917 61 24.4206 60.2499 20.9022 58.7925C17.3837 57.3351 14.1868 55.199 11.4939 52.5061C8.801 49.8132 6.66487 46.6163 5.20749 43.0978C3.7501 39.5794 3 35.8083 3 32C3 28.1917 3.75011 24.4206 5.2075 20.9022C6.66489 17.3837 8.80101 14.1868 11.4939 11.4939C14.1868 8.80099 17.3838 6.66487 20.9022 5.20749C24.4206 3.7501 28.1917 3 32 3L32 3Z"
									stroke="currentColor"
									stroke-width="5"
									stroke-linecap="round"
									stroke-linejoin="round"
								></path>
								<path
									class="text-gray-900"
									d="M32 3C36.5778 3 41.0906 4.08374 45.1692 6.16256C49.2477 8.24138 52.7762 11.2562 55.466 14.9605C58.1558 18.6647 59.9304 22.9531 60.6448 27.4748C61.3591 31.9965 60.9928 36.6232 59.5759 40.9762"
									stroke="currentColor"
									stroke-width="5"
									stroke-linecap="round"
									stroke-linejoin="round"
								>
								</path>
							</svg>
						</div>
					</div>
				@endif

				<div class="mt-4 border-t pt-3 text-end">
					<x-button
						@click.prevent="modalOpen = false"
						variant="outline"
					>
						{{ __('Close') }}
					</x-button>
					<x-button href="{{ route('dashboard.user.payment.subscription') }}">
						{{ __('Upgrade Plan') }}
					</x-button>
				</div>
			</x-slot:modal>
		</x-modal>
	</div>
@endif

<script>
	let creditsLoaded{{ $random }} = false;

	function loadCredits{{ $random }}() {
		if (creditsLoaded{{ $random }}) {
			return;
		}

		fetch('{!! route('credit-list-partial', ['cache_key' => $plan?->id ? 'credit-list-plan-cache' : request('credit-list-cache'), 'plan_id' => $plan?->id, 'user_id' => $user?->id]) !!}')
			.then(response => response.json())
			.then(data => {
				let ID1 = '#credit-list-partial-direct-{{ $random }}';
				let ID2 = '#credit-list-partial-{{ $random }}';

				if (document.querySelector(ID1)) {
					document.querySelector(ID1).innerHTML = data.html;
				}

				if (document.querySelector(ID2)) {
					document.querySelector(ID2).innerHTML = data.html;
				}
			});
	}
</script>
