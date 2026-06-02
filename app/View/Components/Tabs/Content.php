<?php

namespace App\View\Components\Tabs;

use Closure;
use Illuminate\Contracts\View\View;
use Illuminate\View\Component;

class Content extends Component
{
    public function __construct(
        public string $name,
    ) {}

    public function render(): View|Closure|string
    {
        return view('components.tabs.content');
    }
}
