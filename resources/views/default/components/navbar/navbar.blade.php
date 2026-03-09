@php
    $isChatImageRoute = request()->is('ai-image-pro*');
    $themeNavbar = setting('dash_theme', 'default');
    $navbarView = "{$themeNavbar}.components.navbar.navbar";
@endphp

@if ($isChatImageRoute)
    @if ($themeNavbar === 'default')
        @include('ai-image-pro::quick-menu')
    @elseif (View::exists($navbarView))
        @include($navbarView)
    @else
        @include('components.navbar.navbar-default')
    @endif
@else
    @include('components.navbar.navbar-default')
@endif
