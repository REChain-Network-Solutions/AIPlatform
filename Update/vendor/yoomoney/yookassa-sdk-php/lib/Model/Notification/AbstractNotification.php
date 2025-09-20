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

namespace YooKassa\Model\Notification;

use YooKassa\Common\AbstractObject;
use YooKassa\Model\Deal\DealInterface;
use YooKassa\Model\Payment\PaymentInterface;
use YooKassa\Model\Payout\PayoutInterface;
use YooKassa\Model\Refund\RefundInterface;
use YooKassa\Validator\Constraints as Assert;

/**
 * Базовый класс уведомлений.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @example 03-notification.php 3 Пример скрипта обработки уведомления
 *
 * @property string $type Тип уведомления в виде строки
 * @property string $event Тип события
 */
abstract class AbstractNotification extends AbstractObject implements NotificationInterface
{
    /**
     * @var string|null Тип уведомления
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Choice(callback: [NotificationType::class, 'getValidValues'])]
    protected ?string $_type = null;

    /**
     * @var string|null Тип произошедшего события
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Choice(callback: [NotificationEventType::class, 'getValidValues'])]
    protected ?string $_event = null;

    /**
     * Возвращает тип уведомления.
     *
     * Тип уведомления - одна из констант, указанных в перечислении {@link NotificationType}.
     *
     * @return string|null Тип уведомления в виде строки
     */
    public function getType(): ?string
    {
        return $this->_type;
    }

    /**
     * Устанавливает тип уведомления.
     *
     * @param string|null $type Тип уведомления
     *
     * @return self
     */
    protected function setType(?string $type): self
    {
        $this->_type = $this->validatePropertyValue('_type', $type);
        return $this;
    }

    /**
     * Возвращает тип события.
     *
     * Тип события - одна из констант, указанных в перечислении {@link NotificationEventType}.
     *
     * @return string|null Тип события
     */
    public function getEvent(): ?string
    {
        return $this->_event;
    }

    /**
     * Устанавливает тип события.
     *
     * @param string|null $event Тип события
     *
     * @return self
     */
    protected function setEvent(?string $event): self
    {
        $this->_event = $this->validatePropertyValue('_event', $event);
        return $this;
    }

    /**
     * Возвращает объект с информацией о платеже или возврате, уведомление о котором хранится в текущем объекте.
     *
     * Так как нотификация может быть сгенерирована и поставлена в очередь на отправку гораздо раньше, чем она будет
     * получена на сайте, то опираться на статус пришедшего платежа не стоит, лучше запросить текущую информацию о
     * платеже у API.
     *
     * @return PaymentInterface|RefundInterface|PayoutInterface|DealInterface|null Объект с информацией о платеже
     */
    abstract public function getObject(): PaymentInterface|RefundInterface|PayoutInterface|DealInterface|null;
}
