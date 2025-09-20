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
 * Класс, представляющий модель ReceiverDigitalWallet.
 *
 * Реквизиты для пополнения баланса электронного кошелька.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $type Код получателя оплаты.
 * @property string $accountNumber Идентификатор электронного кошелька для пополнения. Максимум 20 символов.
 * @property string $account_number Идентификатор электронного кошелька для пополнения. Максимум 20 символов.
*/
class ReceiverDigitalWallet extends AbstractReceiver
{
    /**
     * Идентификатор электронного кошелька для пополнения. Максимум 20 символов.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Length(max: 20)]
    private ?string $_account_number = null;

    public function __construct(?array $data = [])
    {
        parent::__construct($data);
        $this->setType(ReceiverType::DIGITAL_WALLET);
    }

    /**
     * Возвращает account_number.
     *
     * @return string|null Идентификатор электронного кошелька для пополнения.
     */
    public function getAccountNumber(): ?string
    {
        return $this->_account_number;
    }

    /**
     * Устанавливает account_number.
     *
     * @param string|null $account_number Идентификатор электронного кошелька для пополнения.
     *
     * @return self
     */
    public function setAccountNumber(?string $account_number = null): self
    {
        $this->_account_number = $this->validatePropertyValue('_account_number', $account_number);
        return $this;
    }

}

