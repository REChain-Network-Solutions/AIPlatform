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

namespace YooKassa\Model\Payment\PaymentMethod\B2b\Sberbank;

use YooKassa\Model\AmountInterface;
use YooKassa\Model\MonetaryAmount;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель MixedVatData.
 *
 * Данные об НДС, если создается платеж на несколько товаров или услуг с разными ставками НДС (в параметре `type` передано значение ~`mixed`).
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property AmountInterface $amount Сумма НДС
*/
class MixedVatData extends AbstractVatData
{
    /**
     * @var AmountInterface|null Сумма НДС
     */
    #[Assert\NotBlank]
    #[Assert\Type(MonetaryAmount::class)]
    private ?AmountInterface $_amount = null;

    public function __construct(?array $data = [])
    {
        parent::__construct($data);
        $this->setType(VatDataType::MIXED);
    }

    /**
     * Возвращает сумму НДС
     *
     * @return AmountInterface|null Сумма НДС
     */
    public function getAmount(): ?AmountInterface
    {
        return $this->_amount;
    }

    /**
     * Устанавливает сумму НДС
     *
     * @param null|AmountInterface|array $amount Сумма НДС
     */
    public function setAmount(mixed $amount): self
    {
        $this->_amount = $this->validatePropertyValue('_amount', $amount);
        return $this;
    }
}

