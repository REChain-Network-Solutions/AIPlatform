<?php

namespace Modules\EInvoice\Views\Components\Form;

use Illuminate\View\Component;

class Client extends Component
{
    public $clientDetails;

    /**
     * Create a new component instance.
     *
     * @return void
     */
    public function __construct($clientDetails = null)
    {
        $this->clientDetails = $clientDetails;
    }

    /**
     * Get the view / contents that represent the component.
     *
     * @return \Illuminate\View\View|string
     */
    public function render()
    {
        return view('einvoice::components.form.client');
    }

}
