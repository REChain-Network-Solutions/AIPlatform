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
 * Класс, представляющий модель CalculatedVatData.
 *
 * Данные об НДС, если товар или услуга облагается налогом (в параметре `type` передано значение ~`calculated`).
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $rate Налоговая ставка (в процентах). Возможные значения: ~`5`, ~`7`, ~`10` и ~`20`.
 * @property AmountInterface $amount Сумма НДС
*/
class CalculatedVatData extends AbstractVatData
{
    /**
     * @var string|null Налоговая ставка НДС
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Choice(callback: [VatDataRate::class, 'getValidValues'])]
    private ?string $_rate = null;

    /**
     * @var AmountInterface|null Сумма НДС
     */
    #[Assert\NotBlank]
    #[Assert\Type(MonetaryAmount::class)]
    private ?AmountInterface $_amount = null;

    public function __construct(?array $data = [])
    {
        parent::__construct($data);
        $this->setType(VatDataType::CALCULATED);
    }

    /**
     * Возвращает налоговую ставку НДС
     *
     * @return string|null Налоговая ставка НДС
     */
    public function getRate(): ?string
    {
        return $this->_rate;
    }

    /**
     * Устанавливает налоговую ставку НДС
     *
     * @param string|null $rate Налоговая ставка НДС
     */
    public function setRate(?string $rate): self
    {
        $this->_rate = $this->validatePropertyValue('_rate', $rate);
        return $this;
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

