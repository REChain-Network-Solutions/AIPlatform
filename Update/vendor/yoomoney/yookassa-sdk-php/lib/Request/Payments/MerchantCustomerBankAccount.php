<?php

/*
 * The MIT License
 *
 * Copyright (c) 2025 "YooMoney", NBСO LLC
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

namespace YooKassa\Request\Payments;

use YooKassa\Common\AbstractObject;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель MerchantCustomerBankAccount.
 *
 * Данные банковского счета, открытого в вашей системе. Необходимо передавать, если пользователь %[пополняет свой счет](/developers/payment-acceptance/scenario-extensions/bank-account-in-merchant-system).
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 * @property string $accountNumber Номер банковского счета. Формат — 20 символов.
 * @property string $account_number Номер банковского счета. Формат — 20 символов.
 * @property string $bic Банковский идентификационный код (БИК) банка, в котором открыт счет.
*/
class MerchantCustomerBankAccount extends AbstractObject
{
    /**
     * Номер банковского счета. Формат — 20 символов.
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    #[Assert\Regex("/[0-9]{20}/")]
    private ?string $_account_number = null;

    /**
     * Банковский идентификационный код (БИК) банка, в котором открыт счет.
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    #[Assert\Regex("/\\d{9}/")]
    private ?string $_bic = null;

    /**
     * Возвращает номер банковского счета.
     *
     * @return string|null
     */
    public function getAccountNumber(): ?string
    {
        return $this->_account_number;
    }

    /**
     * Устанавливает номер банковского счета.
     *
     * @param string|null $account_number Номер банковского счета.
     *
     * @return self
     */
    public function setAccountNumber(?string $account_number = null): self
    {
        $this->_account_number = $this->validatePropertyValue('_account_number', $account_number);
        return $this;
    }

    /**
     * Возвращает БИК.
     *
     * @return string|null
     */
    public function getBic(): ?string
    {
        return $this->_bic;
    }

    /**
     * Устанавливает БИК.
     *
     * @param string|null $bic Банковский идентификационный код (БИК) банка, в котором открыт счет.
     *
     * @return self
     */
    public function setBic(?string $bic = null): self
    {
        $this->_bic = $this->validatePropertyValue('_bic', $bic);
        return $this;
    }

}
