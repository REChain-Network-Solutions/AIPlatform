<?php

namespace App\View\Components;

use Closure;
use Illuminate\Contracts\View\View;
use Illuminate\View\Component;

class HeaderSearch extends Component
{
    /**
     * Create a new component instance.
     */
    public function __construct(
        public bool $showIcon = true,
        public bool $showKbd = true,
        public bool $showArrow = true,
        public bool $showLoader = true,
        public bool $outlineGlow = false,
    ) {
        //
    }

    /**
     * Get the view / contents that represent the component.
     */
    public function render(): View|Closure|string
    {
        return view('components.header-search.header-search-screen');
    }
}
