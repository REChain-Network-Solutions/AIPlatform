@if($item['is_buy'])
	@if(isset($item['parent']))
		@if(\App\Helpers\Classes\MarketplaceHelper::isRegistered($item['parent']['slug']))
			@if(in_array($item['slug'], ['whatsapp', 'telegram', 'facebook', 'instagram']))
				@if(\App\Helpers\Classes\MarketplaceHelper::getDbVersion($item['parent']['slug']) >= $item['parent']['min_version'])
					@if($item['only_premium'])
						@if($item['check_subscription'])
							<x-button
								target="_blank"
								class="w-full"
								size="lg"
								href="{{ $item['routes']['payment'] }}"
							>
								{{ __('Buy Now') }}
							</x-button>
						@else
							<x-button
								class="w-full"
								size="lg"
								href="{{ $marketSubscription['extensionPayment'] ?? '' }}"
								onclick="return toastr.info('This extension is for premium customers only.')"
							>
								{{ __('Only Premium Customer') }}
							</x-button>
						@endif
					@else
						<x-button
							target="_blank"
							class="w-full"
							size="lg"
							href="{{ $item['routes']['payment'] }}"
						>
							{{ __('Buy Now') }}
						</x-button>
					@endif

				@else
					<x-button
						class="w-full"
						size="lg"
						href="#"
						onclick="return toastr.info('{{ $item['parent']['min_version_message'] }}')"
					>
						{{ __('Buy Now') }}
					</x-button>
				@endif
			@else
				@if($item['only_premium'])
					@if($item['check_subscription'])
						<x-button
							target="_blank"
							class="w-full"
							size="lg"
							href="{{ $item['routes']['payment'] }}"
						>
							{{ __('Buy Now') }}
						</x-button>
					@else
						<x-button
							class="w-full"
							size="lg"
							href="{{ $marketSubscription['extensionPayment'] ?? '' }}"
							onclick="return toastr.info('This extension is for premium customers only.')"
						>
							{{ __('Only Premium Customer') }}
						</x-button>
					@endif
				@else
					<x-button
						target="_blank"
						class="w-full"
						size="lg"
						href="{{ $item['routes']['payment'] }}"
					>
						{{ __('Buy Now') }}
					</x-button>
				@endif
			@endif
		@else
			<x-button
				class="w-full"
				size="lg"
				href="#"
				onclick="return toastr.info('{{ $item['parent']['message'] }}')"
			>
				{{ __('Buy Now') }}
			</x-button>
		@endif

	@else
		@if($item['only_premium'])
			@if($item['check_subscription'])
				<x-button
					target="_blank"
					class="w-full"
					size="lg"
					href="{{ $item['routes']['payment'] }}"
				>
					{{ __('Buy Now') }}
				</x-button>
			@else
				<x-button
					class="w-full"
					size="lg"
					href="{{ $marketSubscription['extensionPayment'] ?? '' }}"
					onclick="return toastr.info('This extension is for premium customers only.')"
				>
					{{ __('Only Premium Customer') }}
				</x-button>
			@endif
		@else
			<x-button
				target="_blank"
				class="w-full"
				size="lg"
				href="{{ $item['routes']['payment'] }}"
			>
				{{ __('Buy Now') }}
			</x-button>
		@endif
	@endif

@else
	<span
		class="lqd-tooltip-container group relative inline-flex cursor-default before:absolute before:-start-1.5 before:-top-1.5 before:h-7 before:w-7">
                                                    <span class="lqd-tooltip-icon ">
                                                         <x-button
															 target="_blank"
															 class="w-full lqd-tooltip-icon "
															 size="lg"
															 type="button"
														 >
                                                                {{ __('Learn more') }}
                                                            </x-button>
                                                    </span>
                                                    <span
														class="lqd-tooltip-content min-w-64 invisible absolute bottom-full start-1/2 z-50 mb-3 -translate-x-1/2 translate-y-1 rounded-xl bg-background/80 px-4 py-3  text-xs leading-normal text-foreground opacity-0 shadow-lg shadow-black/5 backdrop-blur-sm backdrop-saturate-150 transition-all before:absolute before:inset-x-0 before:-bottom-3 before:h-3 group-hover:visible group-hover:translate-y-0 group-hover:opacity-100"
													>
                                                            Contact us for more information:
                                                           <span class="flex justify-center mt-1">info@liquid-themes.com <x-tabler-copy class="ms-2"
																																		id="copyButton"/></span>

                                                    </span>
                                                </span>
@endif
