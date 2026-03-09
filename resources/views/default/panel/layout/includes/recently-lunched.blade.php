@if (count($recently_launched) > 0)
    <ul>
        @foreach ($recently_launched as $item)
			@if($item && $item->slug)
				<li class="border-b px-3 py-2 transition-colors last:border-b-0 hover:bg-foreground/5">
					<a
						class="flex items-center gap-2 text-heading-foreground"
						href="{{ route('dashboard.user.openai.documents.single', $item->slug) }}"
					>
						<x-lqd-icon
							size="lg"
							style="background: {{ $item->generator->color }}"
						>
							<span class="flex size-5">
								@if ($item->generator->image !== 'none')
									{!! html_entity_decode($item->generator->image) !!}
								@endif
							</span>
						</x-lqd-icon>
						{{ $item->title ?: 'Document' }}
						<small class="ms-auto text-foreground/50">{{ $item->generator->type == 'text' ? __('Document') : __(ucfirst($item->generator->type)) }}</small>
					</a>
				</li>
			@endif
        @endforeach
    </ul>
@endif

@if (isset($recently_launched) and $recently_launched == 'null')
    <div class="block p-6 text-center font-medium text-heading-foreground">
        <h3 class="mb-2">{{ __('There is no recent lunch.') }}</h3>
    </div>
@endif
