<?php

/**
 * Author: Emmanuel Paul Mnzava
 * Twitter: @epmnzava
 * Github:https://github.com/dbrax/income-expense
 * Email: epmnzava@gmail.com
 * 
 */

namespace  Epmnzava\IncomeExpense\Models;

use Illuminate\Database\Eloquent\Model;
use Epmnzava\IncomeExpense\Models\Income;

class IncomeCategory extends Model
{

    protected $guarded = [];
    protected $table = "income_category";

        public function Incomes()
    {
        return $this->hasMany(Income::class);
    }
}
