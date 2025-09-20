<?php

namespace App\View\Components;

use Closure;
use Illuminate\Contracts\View\View;
use Illuminate\View\Component;

class TransactionStatus extends Component
{
    /**
     * Create a new component instance.
     */
    public function __construct(
        public string $status = 'Completed'
    ) {}

    /**
     * Get the view / contents that represent the component.
     */
    public function render(): View|Closure|string
    {
        return view('components.transaction-status');
    }
}
