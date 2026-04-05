@php($protectedPaths = $protectedPaths ?? [])
@if(count($protectedPaths))
<div class="alert alert-light border">
    <strong>Protected paths</strong>
    <ul class="mt-2 mb-0">
        @foreach($protectedPaths as $path)
            <li>{{ $path }}</li>
        @endforeach
    </ul>
</div>
@endif
