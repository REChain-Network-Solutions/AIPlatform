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

use Exception;
use YooKassa\Common\AbstractObject;
use YooKassa\Model\Payment\PaymentMethodStatus;
use YooKassa\Model\Payment\PaymentMethodType;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель AbstractPaymentMethod.
 *
 * Абстрактный класс, описывающий основные свойства и методы платежных методов.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $type Код способа оплаты
 * @property string $id Идентификатор записи о сохраненных платежных данных
 * @property bool $saved Возможность многократного использования
 * @property string $title Название метода оплаты
 * @property string $status Название метода оплаты
 */
abstract class AbstractPaymentMethod extends AbstractObject
{
    /**
     * Код способа оплаты.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Choice(callback: [PaymentMethodType::class, 'getValidValues'])]
    protected ?string $_type = null;

    /**
     * Идентификатор записи о сохраненных платежных данных.
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    protected ?string $_id = null;

    /**
     * Признак сохранения способа оплаты для %[автоплатежей](https://yookassa.ru/developers/payment-acceptance/scenario-extensions/recurring-payments/pay-with-saved).
     * Возможные значения:   * ~`true` — способ оплаты сохранен для автоплатежей и выплат; * ~`false` — способ оплаты не сохранен.
     *
     * @var bool|null
     */
    #[Assert\NotNull]
    #[Assert\Type('bool')]
    protected bool $_saved = false;

    /**
     * Название способа оплаты.
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    protected ?string $_title = null;

    /**
     * Статус проверки и сохранения способа оплаты.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Choice(callback: [PaymentMethodStatus::class, 'getValidValues'])]
    protected ?string $_status = null;

    /**
     * Возвращает тип платежного метода.
     *
     * @return string|null Тип платежного метода
     */
    public function getType(): ?string
    {
        return $this->_type;
    }

    /**
     * Устанавливает тип платежного метода.
     *
     * @param string|null $type Тип платежного метода
     *
     * @return self
     */
    public function setType(?string $type = null): self
    {
        $this->_type = $this->validatePropertyValue('_type', $type);
        return $this;
    }

    /**
     * Возвращает id.
     *
     * @return string|null
     */
    public function getId(): ?string
    {
        return $this->_id;
    }

    /**
     * Устанавливает id.
     *
     * @param string|null $id Идентификатор способа оплаты.
     *
     * @throws Exception
     * @return self
     */
    public function setId(?string $id = null): self
    {
        $this->_id = $this->validatePropertyValue('_id', $id);
        return $this;
    }

    /**
     * Возвращает saved.
     *
     * @return bool|null
     */
    public function getSaved(): ?bool
    {
        return $this->_saved;
    }

    /**
     * Устанавливает признак возможности многократного использования.
     *
     * @param bool|null $saved Признак сохранения способа оплаты для %[автоплатежей](https://yookassa.ru/developers/payment-acceptance/scenario-extensions/recurring-payments/pay-with-saved).
     * Возможные значения: * ~`true` — способ оплаты сохранен для автоплатежей и выплат; * ~`false` — способ оплаты не сохранен.
     *
     * @return self
     * @throws Exception
     */
    public function setSaved(mixed $saved = null): self
    {
        $this->_saved = $this->validatePropertyValue('_saved', $saved);
        return $this;
    }

    /**
     * Возвращает status.
     *
     * @return string|null
     */
    public function getStatus(): ?string
    {
        return $this->_status;
    }

    /**
     * Устанавливает status.
     *
     * @param string|null $status
     *
     * @return self
     */
    public function setStatus(?string $status = null): self
    {
        $this->_status = $this->validatePropertyValue('_status', $status);
        return $this;
    }

    /**
     * Возвращает Название способа оплаты.
     *
     * @return string|null Название способа оплаты
     */
    public function getTitle(): ?string
    {
        return $this->_title;
    }

    /**
     * Устанавливает Название способа оплаты.
     *
     * @param string|null $title Название способа оплаты.
     *
     * @throws Exception
     * @return self
     */
    public function setTitle(?string $title = null): self
    {
        $this->_title = $this->validatePropertyValue('_title', $title);
        return $this;
    }
}
