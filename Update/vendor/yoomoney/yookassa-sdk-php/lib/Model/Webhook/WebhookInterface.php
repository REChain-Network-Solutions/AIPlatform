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

/**
 * Interface WebhookInterface.
 *
 * @category Interface
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $id Идентификатор webhook
 * @property string $event Событие, о котором уведомляет ЮKassa
 * @property string $url URL, на который ЮKassa будет отправлять уведомления
 */
interface WebhookInterface
{
    /**
     * Возвращает идентификатор webhook.
     *
     * @return string|null Идентификатор webhook
     */
    public function getId(): ?string;

    /**
     * Устанавливает идентификатор webhook.
     *
     * @param string|null $id Идентификатор webhook
     *
     * @return self
     */
    public function setId(?string $id = null): self;

    /**
     * Возвращает событие, о котором уведомляет ЮKassa.
     *
     * @return string|null Событие, о котором уведомляет ЮKassa
     */
    public function getEvent(): ?string;

    /**
     * Устанавливает событие, о котором уведомляет ЮKassa.
     *
     * @param string|null $event Событие, о котором уведомляет ЮKassa
     *
     * @return self
     */
    public function setEvent(?string $event = null): self;

    /**
     * Возвращает URL, на который ЮKassa будет отправлять уведомления.
     *
     * @return string|null URL, на который ЮKassa будет отправлять уведомления
     */
    public function getUrl(): ?string;

    /**
     * Устанавливает URL, на который ЮKassa будет отправлять уведомления.
     *
     * @param string|null $url URL, на который ЮKassa будет отправлять уведомления
     *
     * @return self
     */
    public function setUrl(?string $url = null): self;
}
