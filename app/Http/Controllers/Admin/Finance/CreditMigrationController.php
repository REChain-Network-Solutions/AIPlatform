<?php

declare(strict_types=1);

namespace App\Http\Controllers\Admin\Finance;

use App\Http\Controllers\Controller;
use Illuminate\View\View;

class CreditMigrationController extends Controller
{
    public function index(): View
    {
        return view('panel.admin.finance.credit-migration.index');
    }
}
