<div class="text-center p-5 border rounded bg-light">
  <h5 class="mb-2">{{ $title ?? 'Nothing here yet' }}</h5>
  <p class="text-muted mb-3">{{ $slot }}</p>
  @if(!empty($cta))
    <a href="{{ $cta['href'] }}" class="btn btn-outline-primary">{{ $cta['label'] }}</a>
  @endif
</div>
