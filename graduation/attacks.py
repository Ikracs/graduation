import os
import sys
import time
import math
import torch
import numpy as np

import utils
from imagenet_labels import label_to_name

def zeroth_order_opt(model, img, label, config, timestamp):
    budget    = config['budget']
    epsilon   = config['epsilon']
    
    log_freq  = config['log_freq']
    log_root  = config['log_root']
    save_root = config['save_root']
    
    delta     = config.get('delta', 0.1)
    alpha     = config.get('alpha', 1e-2)
    n_samples = config.get('n_samples', 20)
    momentum  = config.get('momentum', 0.5)

    print('ZOO with ')
    print('delta: {}'.format(delta))
    print('alpha: {}'.format(alpha))
    print('n_smaples: {}'.format(n_samples))
    print('momentum: {}'.format(momentum))

    logger = utils.Logger()

    min_v, max_v = 0, 1 if img.max() <= 1 else 255

    def _normalize(x):
        flatten = x.reshape(x.shape[0], -1)
        l2_norm = np.linalg.norm(flatten, axis=-1, keepdims=True)
        return (flatten / l2_norm).reshape(*x.shape)
    
    def _project(o, x):
        x = o + np.clip(x - o, -epsilon, epsilon)
        return x.clip(min_v, max_v)

    adv_img = np.copy(img)
    grad = np.zeros_like(img)
    query = np.ones(img.shape[0])

    logits = model.predict(adv_img)
    loss = model.loss(logits, label)
    done = model.done(logits, label)

    start_time = time.time()
    max_iter = budget // (n_samples + 1)
    print('attacking correct classified images...')
    for i_iter in range(max_iter):
        if done.sum() == img.shape[0]: break

        grad_new = np.zeros_like(adv_img[~done])
        for _ in range(n_samples):
            basis = _normalize(np.random.randn(*adv_img[~done].shape))
            adv_img_delta = _project(img[~done], adv_img[~done] + delta * basis)
            
            logits = model.predict(adv_img_delta)
            loss_delta = model.loss(logits, label[~done])
            done_delta = model.done(logits, label[~done])

            diff = (loss_delta - loss[~done]) / delta
            grad_new += diff.reshape(-1, 1, 1, 1) * basis
        grad[~done] = momentum * grad[~done] + grad_new
        adv_img[~done] = _project(img[~done], adv_img[~done] + alpha * np.sign(grad[~done]))

        logits = model.predict(adv_img[~done])
        loss[~done] = model.loss(logits, label[~done])
        done[~done] = model.done(logits, label[~done])
        query[~done] += n_samples + 1
        
        if i_iter % log_freq == 0 and done.sum().item() > 0:
            print('[Iter  {:0>4d}] '.format(i_iter), end='')
            print('Success rate: {:.3f}, '.format(done.mean()), end='')
            print('Avg. query: {:.2f}, '.format(query[done].mean()), end='')
            print('Med. query: {:.0f}'.format(np.median(query[done])))
            logger.log_info(done.mean(), query[done].mean())
    
    if log_root is not None:
        save_name = '-'.join(timestamp.split(' '))
        logger.save(os.path.join(log_root, save_name))
    
    if save_root is not None:
        ori_root = os.path.join(save_root, 'clean')
        save_name = [label_to_name(l) for l in label[done]]
        utils.save_imgs(img[done], save_name, ori_root)

        adv_root = os.path.join(save_root, 'adv')
        adv_label = model.predict(adv_img[done]).argmax(axis=-1)
        save_name = [
            '#'.join([label_to_name(l), label_to_name(adv_l)])
            for l, adv_l in zip(label[done], adv_label)
        ]
        utils.save_imgs(adv_img[done], save_name, adv_root)
    
    print('Final results: ', end='')
    print('Success rate: {:.3f}, Avg. query: {:.2f}, Med. query: {:.0f}'.format(
        done.mean(), query[done].mean(), np.median(query[done])))
    
    elapsed = time.time() - start_time
    hours = elapsed // 3600; minutes = (elapsed % 3600) // 60
    print('Time Elapsed: {:.0f} h {:.0f} m'.format(hours, minutes))

def natural_evolution_strategy(model, img, label, config, timestamp):
    budget    = config['budget']
    epsilon   = config['epsilon']
    
    log_freq  = config['log_freq']
    log_root  = config['log_root']
    save_root = config['save_root']
    
    sigma     = config.get('sigma', 0.1)
    alpha     = config.get('alpha', 1e-2)
    n_samples = config.get('n_samples', 10)
    momentum  = config.get('momentum', 0.5)

    print('NES with ')
    print('sigma: {}'.format(sigma))
    print('alpha: {}'.format(alpha))
    print('n_smaples: {}'.format(n_samples))
    print('momentum: {}'.format(momentum))
    
    logger = utils.Logger()

    min_v, max_v = 0, 1 if img.max() <= 1 else 255

    def _project(o, x):
        x = o + np.clip(x - o, -epsilon, epsilon)
        return x.clip(min_v, max_v)

    adv_img = np.copy(img)
    grad = np.zeros_like(img)
    query = np.ones(img.shape[0])

    logits = model.predict(adv_img)
    loss = model.loss(logits, label)
    done = model.done(logits, label)

    start_time = time.time()
    max_iter = budget // (2 * n_samples + 1)
    print('attacking correct classified images...')
    for i_iter in range(max_iter):
        if done.sum() == img.shape[0]: break

        grad_new = np.zeros_like(adv_img[~done])
        for _ in range(n_samples):
            sample = np.random.randn(*adv_img[~done].shape)
            adv_img_pos = _project(img[~done], adv_img[~done] + sigma * sample)
            adv_img_neg = _project(img[~done], adv_img[~done] - sigma * sample)
            
            logits = model.predict(adv_img_pos)
            loss_pos = model.loss(logits, label[~done])
            done_pos = model.done(logits, label[~done])

            logits = model.predict(adv_img_neg)
            loss_neg = model.loss(logits, label[~done])
            done_neg = model.done(logits, label[~done])

            loss_sum = (loss_pos - loss_neg).reshape(-1, 1, 1, 1)
            grad_new += (loss_sum * sample) / (2 * sigma)
        grad[~done] = momentum * grad[~done] + grad_new
        adv_img[~done] = _project(img[~done], adv_img[~done] + alpha * np.sign(grad[~done]))

        logits = model.predict(adv_img[~done])
        loss[~done] = model.loss(logits, label[~done])
        done[~done] = model.done(logits, label[~done])
        query[~done] += 2 * n_samples + 1
        
        if i_iter % log_freq == 0 and done.sum().item() > 0:
            print('[Iter  {:0>4d}] '.format(i_iter), end='')
            print('Success rate: {:.3f}, '.format(done.mean()), end='')
            print('Avg. query: {:.2f}, '.format(query[done].mean()), end='')
            print('Med. query: {:.0f}'.format(np.median(query[done])))
            logger.log_info(done.mean(), query[done].mean())
    
    if log_root is not None:
        save_name = '-'.join(timestamp.split(' '))
        logger.save(os.path.join(log_root, save_name))
    
    if save_root is not None:
        ori_root = os.path.join(save_root, 'clean')
        save_name = [label_to_name(l) for l in label[done]]
        utils.save_imgs(img[done], save_name, ori_root)

        adv_root = os.path.join(save_root, 'adv')
        adv_label = model.predict(adv_img[done]).argmax(axis=-1)
        save_name = [
            '#'.join([label_to_name(l), label_to_name(adv_l)])
            for l, adv_l in zip(label[done], adv_label)
        ]
        utils.save_imgs(adv_img[done], save_name, adv_root)
    
    print('Final results: ', end='')
    print('Success rate: {:.3f}, Avg. query: {:.2f}, Med. query: {:.0f}'.format(
        done.mean(), query[done].mean(), np.median(query[done])))
    
    elapsed = time.time() - start_time
    hours = elapsed // 3600; minutes = (elapsed % 3600) // 60
    print('Time Elapsed: {:.0f} h {:.0f} m'.format(hours, minutes))

def sign_hunter(model, img, label, config, timestamp):
    budget    = config['budget']
    epsilon   = config['epsilon']
    
    log_freq  = config['log_freq']
    log_root  = config['log_root']
    save_root = config['save_root']

    logger = utils.Logger()

    min_v, max_v = 0, 1 if img.max() <= 1 else 255
    
    dim = np.prod(img.shape[1:])
    sign_bits = np.ones((img.shape[0], dim))
    query = np.ones(img.shape[0])
    
    delta = epsilon * sign_bits.reshape(img.shape)
    perturbed = np.clip(img + delta, min_v, max_v)
    
    logits = model.predict(perturbed)
    loss = model.loss(logits, label)
    done = model.done(logits, label)

    node_i, tree_h = 0, 0
    rs = math.ceil(dim / (2 ** tree_h))

    start_time = time.time()
    print('Strat attacking correct classified images...')
    for i_iter in range(budget):
        if (rs < 1) or (done.sum() == img.shape[0]): break

        sign_bits_new = sign_bits.copy()[~done]
        sign_bits_new[:, node_i * rs: (node_i + 1) * rs] *= -1
        
        delta = epsilon * sign_bits_new.reshape(img[~done].shape)
        perturbed = np.clip(img[~done] + delta, min_v, max_v)

        logits = model.predict(perturbed)
        loss_new = model.loss(logits, label[~done])
        done_new = model.done(logits, label[~done])
        query[~done] += 1

        improved = (loss_new > loss[~done])
        loss[~done] = improved * loss_new + ~improved * loss[~done]
        sign_bits[~done] = improved[:, np.newaxis] * sign_bits_new + \
            ~improved[:, np.newaxis] * sign_bits[~done]
        done[~done] = done_new
        
        node_i += 1
        if node_i == 2 ** tree_h:
            node_i = 0; tree_h += 1
            rs = math.ceil(dim / (2 ** tree_h))

        if i_iter % log_freq == 0 and done.sum().item() > 0:
            print('[Iter  {:0>4d}] '.format(i_iter), end='')
            print('Success rate: {:.3f}, '.format(done.mean()), end='')
            print('Avg. query: {:.2f}, '.format(query[done].mean()), end='')
            print('Med. query: {:.0f}'.format(np.median(query[done])))
            logger.log_info(done.mean(), query[done].mean())
    
    if log_root is not None:
        save_name = '-'.join(timestamp.split(' '))
        logger.save(os.path.join(log_root, save_name))
    
    if save_root is not None:
        ori_root = os.path.join(save_root, 'clean')
        save_name = [label_to_name(l) for l in label[done]]
        utils.save_imgs(img[done], save_name, ori_root)

        adv_root = os.path.join(save_root, 'adv')
        delta = epsilon * sign_bits[done].reshape(img[done].shape)
        perturbed = np.clip(img[done] + delta, min_v, max_v)
        adv_label = model.predict(perturbed).argmax(axis=-1)
        save_name = [
            '#'.join([label_to_name(l), label_to_name(adv_l)])
            for l, adv_l in zip(label[done], adv_label)
        ]
        utils.save_imgs(perturbed, save_name, adv_root)
    
    print('Final results: ', end='')
    print('Success rate: {:.3f}, Avg. query: {:.2f}, Med. query: {:.0f}'.format(
        done.mean(), query[done].mean(), np.median(query[done])))
    
    elapsed = time.time() - start_time
    hours = elapsed // 3600; minutes = (elapsed % 3600) // 60
    print('Time Elapsed: {:.0f} h {:.0f} m'.format(hours, minutes))

def accelerated_sign_hunter(model, imgs, labels, config, timestamp):
    budget    = config['budget']
    log_freq  = config['log_freq']
    log_root  = config['log_root']
    save_root = config['save_root']

    logger = utils.Logger()
    
    total_num_correct = 0
    total_num_queries = []
    ori_imgs, ori_labels = [], []
    adv_imgs, adv_labels = [], []

    start_time = time.time()
    print('Start attacking correct classified images...')
    for count, (img, label) in enumerate(zip(imgs, labels)):
        img, label = img[np.newaxis, ...], label[np.newaxis, ...]
        done, adv_img, queries = _ash_single_img(model, img, label, config)
        adv_label = model.predict(adv_img).argmax(axis=-1).item()
        
        print('Attack on {:d}th img starts, '.format(count), end='')
        print('original class: {:s}. '.format(label_to_name(label.item())), end='')
        
        if done:    # attack succeeds
            total_num_correct += 1
            total_num_queries.append(queries)
            ori_imgs.append(img)
            ori_labels.append(label.item())
            adv_imgs.append(adv_img)
            adv_labels.append(adv_label)
            
            print('Attack succeeds, ', end='')
            print('final class: {:s}, '.format(label_to_name(adv_label)), end='')
            print('num queries: {:d}'.format(queries))
        else:       # attack fails
            print('Attack fails, ', end='')
            print('final class: {:s}'.format(label_to_name(adv_label)))
    
    ori_imgs, ori_labels = np.concatenate(ori_imgs), np.array(ori_labels)
    adv_imgs, adv_labels = np.concatenate(adv_imgs), np.array(adv_labels)
    
    if log_root is not None:
        total_num_queries = np.array(total_num_queries)
        for i_iter in range(0, budget, log_freq):
            succeed = total_num_queries < i_iter
            if succeed.sum() != 0:
                logger.log_info(
                    succeed.sum() / imgs.shape[0],
                    total_num_queries[succeed].mean()
                )
            else: logger.log_info(0.0, 0)
        save_name = '-'.join(timestamp.split(' '))
        logger.save(os.path.join(log_root, save_name))
    
    if save_root is not None:
        ori_root = os.path.join(save_root, 'clean')
        save_names = [label_to_name(l) for l in ori_labels]
        utils.save_imgs(ori_imgs, save_names, ori_root)
        
        adv_root = os.path.join(save_root, 'adv')
        save_names = [
            '&'.join([label_to_name(l), label_to_name(adv_l)])
            for l, adv_l in zip(ori_labels, adv_labels)
        ]
        utils.save_imgs(adv_imgs, save_names, adv_root)
    
    print('Final results: ', end='')
    print('Success rate: {:.3f}, Avg. query: {:.2f}, Med. query: {:.0f}'.\
        format(
            total_num_correct / imgs.shape[0],
            np.mean(total_num_queries),
            np.median(total_num_queries)
        )
    )
    
    elapsed = time.time() - start_time
    hours = elapsed // 3600; minutes = (elapsed % 3600) // 60
    print('Time Elapsed: {:.0f} h {:.0f} m'.format(hours, minutes))


def _ash_single_img(model, img, label, config):
    budget    = config['budget']
    epsilon   = config['epsilon']
    
    min_v, max_v = 0, 1 if img.max() <= 1 else 255

    dim = np.prod(img.shape)
    sign_bits = np.ones(dim)
    num_queries = 1

    delta = epsilon * sign_bits.reshape(img.shape)
    perturbed = np.clip(img + delta, min_v, max_v)
    
    logits = model.predict(perturbed)
    loss = model.loss(logits, label)
    done = model.done(logits, label)

    node_i, tree_h = 0, 0
    regions = [[0.0, [0, dim]]]

    def _divide(regions):
        regions_new = []
        for region in regions:
            start, end = region[1]
            mid = start + (end - start) // 2
            regions_new.append([region[0], [start, mid]])
            regions_new.append([region[0], [mid, end]])
        return regions_new
    
    while num_queries < budget and not done:
        need_query = True
        if node_i % 2 == 1:
            regions[node_i][0] -= regions[node_i - 1][0]
            need_query = regions[node_i][0] < 0
        
        if need_query:
            sign_bits_new = sign_bits.copy()
            start, end = regions[node_i][1]
            sign_bits_new[start: end] *= -1

            if start != end:
                delta = epsilon * sign_bits_new.reshape(img.shape)
                perturbed = np.clip(img + delta, min_v, max_v)

                logits = model.predict(perturbed)
                loss_new = model.loss(logits, label)
                done = model.done(logits, label)
                num_queries += 1

                regions[node_i][0] = (loss - loss_new).item()

                if loss_new > loss:
                    loss, sign_bits = loss_new, sign_bits_new
            else:
                regions[node_i][0] = float('inf')

        node_i += 1
        if node_i == 2 ** tree_h:
            node_i = 0; tree_h += 1
            
            regions = [[abs(r[0]), r[1]] for r in _divide(regions)]
            regions = sorted(regions, key=lambda r: r[0], reverse=False)
    
    delta = epsilon * sign_bits.reshape(img.shape)
    adv_img = np.clip(img + delta, min_v, max_v)
    return done, adv_img, num_queries
