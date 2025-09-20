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

namespace YooKassa\Model\Webhook;

use YooKassa\Common\AbstractObject;
use YooKassa\Model\Notification\NotificationEventType;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс Webhook содержит информацию о подписке на одно событие.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $id Идентификатор webhook
 * @property string $event Событие, о котором уведомляет ЮKassa
 * @property string $url URL, на который ЮKassa будет отправлять уведомления
 */
class Webhook extends AbstractObject implements WebhookInterface
{
    /**
     * Идентификатор webhook.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    private ?string $_id = null;

    /**
     * Событие, о котором уведомляет ЮKassa.
     *
     * @see NotificationEventType
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Choice(callback: [NotificationEventType::class, 'getValidValues'])]
    private ?string $_event = null;

    /**
     * URL, на который ЮKassa отправляет уведомления.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    private ?string $_url = null;

    /**
     * Возвращает идентификатор webhook.
     *
     * @return string|null Идентификатор webhook
     */
    public function getId(): ?string
    {
        return $this->_id;
    }

    /**
     * Устанавливает идентификатор webhook.
     *
     * @param string|null $id Идентификатор webhook
     *
     * @return self
     */
    public function setId(?string $id = null): self
    {
        $this->_id = $this->validatePropertyValue('_id', $id);
        return $this;
    }

    /**
     * Возвращает событие, о котором уведомляет ЮKassa.
     *
     * @return string|null Событие, о котором уведомляет ЮKassa
     */
    public function getEvent(): ?string
    {
        return $this->_event;
    }

    /**
     * Устанавливает событие, о котором уведомляет ЮKassa.
     *
     * @param string|null $event Событие, о котором уведомляет ЮKassa
     *
     * @return self
     */
    public function setEvent(?string $event = null): self
    {
        $this->_event = $this->validatePropertyValue('_event', $event);
        return $this;
    }

    /**
     * Возвращает URL, на который ЮKassa будет отправлять уведомления.
     *
     * @return string|null URL, на который ЮKassa будет отправлять уведомления
     */
    public function getUrl(): ?string
    {
        return $this->_url;
    }

    /**
     * Устанавливает URL, на который ЮKassa будет отправлять уведомления.
     *
     * @param string|null $url URL, на который ЮKassa будет отправлять уведомления
     *
     * @return self
     */
    public function setUrl(?string $url = null): self
    {
        $this->_url = $this->validatePropertyValue('_url', $url);
        return $this;
    }
}
