<?php

namespace App\View\Components\Tabs;

use Closure;
use Illuminate\Contracts\View\View;
use Illuminate\View\Component;

class Contents extends Component
{
    public bool $insideContents = true;

    public function render(): View|Closure|string
    {
        return view('components.tabs.contents');
    }
}
