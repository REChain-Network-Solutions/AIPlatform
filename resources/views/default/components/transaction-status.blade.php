@php
	$base_class = 'inline-flex items-center leading-none text-base font-semibold leading-snug rounded-md';

	$color = 'var(--foreground)';
	if ($status == 'Pending') {
		$color = '#B58500';
	} elseif ($status == 'Completed') {
		$color = '#118C60';
	}
@endphp

<span {{ $attributes->withoutTwMergeClasses()->twMerge($base_class, $attributes->get('class')) }}
    style="color: {{ $color }}">
    {{ $status }}
</span>
