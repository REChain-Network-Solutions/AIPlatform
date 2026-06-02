<?php

namespace App\View\Components\Tabs;

use Closure;
use Illuminate\Contracts\View\View;
use Illuminate\View\Component;

class Nav extends Component
{
    public function __construct() {}

    public function render(): View|Closure|string
    {
        return view('components.tabs.nav');
    }
}
