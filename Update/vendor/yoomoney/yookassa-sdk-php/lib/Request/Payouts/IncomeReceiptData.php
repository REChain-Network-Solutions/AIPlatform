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

namespace YooKassa\Request\Payouts;

use YooKassa\Common\AbstractObject;
use YooKassa\Model\AmountInterface;
use YooKassa\Model\MonetaryAmount;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель IncomeReceiptData.
 *
 * Данные для формирования чека в сервисе Мой налог. Необходимо передавать,
 * если вы делаете выплату [самозанятому](/developers/payouts/scenario-extensions/self-employed). Только для обычных выплат.
 *
 * @category Class
 * @package  YooKassa\Request
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 * @property string $serviceName Описание услуги, оказанной получателем выплаты. Не более 50 символов
 * @property string $service_name Описание услуги, оказанной получателем выплаты. Не более 50 символов
 * @property AmountInterface $amount Сумма для печати в чеке
 */
class IncomeReceiptData extends AbstractObject
{
    /** @var int Максимальная длина описание услуги */
    public const MAX_LENGTH_SERVICE_NAME = 50;

    /** @var int Минимальная длина описание услуги */
    public const MIN_LENGTH_SERVICE_NAME = 1;

    /**
     * Описание услуги, оказанной получателем выплаты. Не более 50 символов.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Length(max: self::MAX_LENGTH_SERVICE_NAME)]
    #[Assert\Length(min: self::MIN_LENGTH_SERVICE_NAME)]
    private ?string $_service_name = null;

    /**
     * Сумма для печати в чеке.
     * Используется, если сумма в чеке отличается от суммы выплаты. Сумма чека должна быть больше суммы выплаты или равна ей.
     *
     * @var AmountInterface|null
     */
    #[Assert\Valid]
    #[Assert\Type(MonetaryAmount::class)]
    private ?AmountInterface $_amount = null;

    /**
     * Возвращает описание услуги, оказанной получателем выплаты.
     *
     * @return string|null Описание услуги, оказанной получателем выплаты
     */
    public function getServiceName(): ?string
    {
        return $this->_service_name;
    }

    /**
     * Устанавливает описание услуги, оказанной получателем выплаты.
     *
     * @param string|null $service_name Описание услуги, оказанной получателем выплаты
     *
     * @return self
     */
    public function setServiceName(?string $service_name = null): self
    {
        $this->_service_name = $this->validatePropertyValue('_service_name', $service_name);
        return $this;
    }

    /**
     * Возвращает сумму для печати в чеке.
     *
     * @return AmountInterface|null Сумма для печати в чеке
     */
    public function getAmount(): ?AmountInterface
    {
        return $this->_amount;
    }

    /**
     * Устанавливает сумму для печати в чеке.
     *
     * @param AmountInterface|array|null $amount Сумма для печати в чеке
     *
     * @return self
     */
    public function setAmount(mixed $amount = null): self
    {
        $this->_amount = $this->validatePropertyValue('_amount', $amount);
        return $this;
    }
}
