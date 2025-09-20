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

namespace YooKassa\Model\Payment\PaymentMethod;

use YooKassa\Model\Payment\PaymentMethodType;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель PaymentMethodBankCard.
 *
 * Оплата банковской картой.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $card Данные банковской карты
 */
class PaymentMethodBankCard extends AbstractPaymentMethod
{
    /**
     * @var BankCard|null Данные банковской карты
     */
    #[Assert\Valid]
    #[Assert\Type(BankCard::class)]
    protected ?BankCard $_card = null;

    public function __construct(?array $data = [])
    {
        parent::__construct($data);
        $this->setType(PaymentMethodType::BANK_CARD);
    }

    /**
     * Возвращает данные банковской карты.
     *
     * @return BankCard|null Данные банковской карты
     */
    public function getCard(): ?BankCard
    {
        return $this->_card;
    }

    /**
     * Устанавливает данные банковской карты.
     *
     * @param BankCard|array|null $card Данные банковской карты
     *
     * @return self
     */
    public function setCard(mixed $card): self
    {
        $this->_card = $this->validatePropertyValue('_card', $card);
        return $this;
    }
}
