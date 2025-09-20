<div
	class="lqd-posts-list lqd-docs-list group-[&[data-view-mode=grid]]:grid group-[&[data-view-mode=grid]]:grid-cols-2 group-[&[data-view-mode=grid]]:gap-5 md:group-[&[data-view-mode=grid]]:grid-cols-3 lg:group-[&[data-view-mode=grid]]:grid-cols-4 lg:group-[&[data-view-mode=grid]]:gap-8 xl:group-[&[data-view-mode=grid]]:grid-cols-5"
	id="lqd-docs-list"
>
	@php
		$folders = auth()->user()->folders()->get();
	@endphp
	@foreach ($items as $entry)
		<x-documents.item
			:$entry
			style="extended"
			:$folders
		/>
	@endforeach
</div>
