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

namespace YooKassa\Request\Payments\ReceiverData;

use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель ReceiverBankAccount.
 *
 * Реквизиты для пополнения банковского счета.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $type Код получателя оплаты.
 * @property string $accountNumber Номер банковского счета. Формат — 20 символов.
 * @property string $account_number Номер банковского счета. Формат — 20 символов.
 * @property string $bic Банковский идентификационный код (БИК) банка, в котором открыт счет. Формат — 9 символов.
*/
class ReceiverBankAccount extends AbstractReceiver
{
    /**
     * Номер банковского счета. Формат — 20 символов.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Length(max: 20)]
    #[Assert\Regex('/^[0-9]{20}$/')]
    private ?string $_account_number = null;

    /**
     * Банковский идентификационный код (БИК) банка, в котором открыт счет. Формат — 9 символов.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Length(max: 9)]
    #[Assert\Regex('/^[0-9]{9}$/')]
    private ?string $_bic = null;

    public function __construct(?array $data = [])
    {
        parent::__construct($data);
        $this->setType(ReceiverType::BANK_ACCOUNT);
    }

    /**
     * Возвращает account_number.
     *
     * @return string|null Номер банковского счета.
     */
    public function getAccountNumber(): ?string
    {
        return $this->_account_number;
    }

    /**
     * Устанавливает account_number.
     *
     * @param string|null $account_number Номер банковского счета. Формат — 20 символов.
     *
     * @return self
     */
    public function setAccountNumber(?string $account_number = null): self
    {
        $this->_account_number = $this->validatePropertyValue('_account_number', $account_number);
        return $this;
    }

    /**
     * Возвращает bic.
     *
     * @return string|null Банковский идентификационный код (БИК) банка, в котором открыт счет.
     */
    public function getBic(): ?string
    {
        return $this->_bic;
    }

    /**
     * Устанавливает bic.
     *
     * @param string|null $bic Банковский идентификационный код (БИК) банка, в котором открыт счет. Формат — 9 символов.
     *
     * @return self
     */
    public function setBic(?string $bic = null): self
    {
        $this->_bic = $this->validatePropertyValue('_bic', $bic);
        return $this;
    }

}

