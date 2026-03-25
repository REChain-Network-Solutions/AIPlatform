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
use Epmnzava\IncomeExpense\Models\Expense;


class ExpenseCategory extends Model
{

    protected $guarded = [];
    protected $table = "expense_category";

      public function expenses()
    {
        return $this->hasMany(Expense::class);
    }
}
